import unittest

import torch

from src.gcd.infrastructure.xai.methods.cam_methods import GradCamMethod, ScoreCamMethod
from src.gcd.infrastructure.xai.tiling.tile_collector import (
    TileCollectionRequest,
    TileCollector,
)
from src.gcd.infrastructure.xai.tiling.tile_strategy import (
    SlidingWindowTileStrategy,
)


class TileCollectorTests(unittest.TestCase):
    def test_scorecam_forwards_original_then_each_mask_and_uses_the_scores(self) -> None:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature = torch.nn.Conv3d(1, 2, kernel_size=1, bias=False)
                self.head = torch.nn.Conv3d(2, 2, kernel_size=1, bias=False)
                with torch.no_grad():
                    self.feature.weight[:, 0, 0, 0, 0] = torch.tensor([1.0, -1.0])
                    self.head.weight.zero_()
                    self.head.weight[1, 0, 0, 0, 0] = 1.0
                self.xai_layer_targets = {"feature": "feature"}
                self.inputs = []

            def forward(self, value):
                self.inputs.append(value.detach().clone())
                return self.head(self.feature(value))

        model = _Model()
        image = torch.linspace(-1, 1, 8).reshape(1, 2, 2, 2)
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(2, 2, 2), patch_size=2, stride=1
        )
        messages = []
        result = TileCollector().collect(TileCollectionRequest(
            model=model,
            device=torch.device("cpu"),
            model_input=image,
            method=ScoreCamMethod(lambda logits, target: logits[0, target].sum()),
            objective=lambda logits, target: logits[0, target].sum(),
            target_class=1,
            tile_plan=plan,
            method_params={
                "_selected_layer": "feature",
                "_feature_start": 0,
                "_feature_stop": 2,
                "_retain_patch_payload": True,
            },
            logger=messages.append,
        ))

        payload = result.patches[0]
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(payload["masked_forward_count"], 2)
        self.assertIn("masked forwards=2", messages[0])
        self.assertTrue(torch.equal(model.inputs[0], image.unsqueeze(0)))
        self.assertFalse(torch.equal(model.inputs[0], model.inputs[1]))
        self.assertFalse(torch.equal(model.inputs[1], model.inputs[2]))
        self.assertEqual(payload["valid_channels"].tolist(), [True, True])
        activation = payload["layers"]["feature"]["activation"]
        weights = torch.softmax(payload["scores"], dim=0).reshape(1, 2, 1, 1, 1)
        self.assertTrue(torch.allclose(payload["weights"], weights.flatten()))
        expected = (activation * weights).sum(dim=1, keepdim=True).clamp_min(0)
        self.assertTrue(torch.allclose(payload["cam"], expected))

    def test_scorecam_uses_full_volume_prediction_as_fixed_tile_target(self) -> None:
        class _ContextModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature = torch.nn.Identity()
                self.xai_layer_targets = {"feature": "feature"}
                self.forward_count = 0

            def forward(self, value):
                self.forward_count += 1
                feature = self.feature(value)
                threshold = feature.mean(dim=(2, 3, 4), keepdim=True) + 0.05
                return torch.cat([torch.zeros_like(feature), feature - threshold], dim=1)

        model = _ContextModel()
        image = torch.tensor([0.1, 0.4, 0.7]).view(1, 3, 1, 1).expand(1, 3, 2, 2)
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(3, 2, 2), patch_size=2, stride=1
        )
        collection = TileCollector().collect(TileCollectionRequest(
            model=model,
            device=torch.device("cpu"),
            model_input=image,
            method=ScoreCamMethod(lambda logits, target: logits[0, target].sum()),
            objective=lambda logits, target: logits[0, target].sum(),
            target_class=1,
            tile_plan=plan,
            method_params={"_selected_layer": "feature", "_objective_id": "predicted_target_mask"},
        ))
        self.assertEqual(model.forward_count, 2 * len(plan.regions) + 1)
        self.assertEqual([p["target_present"] for p in collection.patches], [False, True])
        self.assertEqual([p["masked_forward_count"] for p in collection.patches], [0, 1])
        # The first window predicts class 1 at its right edge, but the
        # full-volume blended prediction does not.
        self.assertTrue((collection.patches[0]["pred"].argmax(dim=1)[0, 1] == 1).all())

    def test_scorecam_uses_hidden_decoder_output_channel_count(self) -> None:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature = torch.nn.Conv3d(1, 3, kernel_size=1)
                self.head = torch.nn.Conv3d(3, 2, kernel_size=1)
                self.xai_layer_targets = {
                    "decoder 1": "head",
                    "decoder 2": "feature",
                }

            def forward(self, value):
                return self.head(self.feature(value))

        plan = SlidingWindowTileStrategy().plan(
            input_shape=(2, 2, 2), patch_size=2, stride=0
        )
        for layer, expected_count in (("decoder 2", 3),):
            with self.subTest(layer=layer):
                result = TileCollector().collect(TileCollectionRequest(
                    model=_Model(),
                    device=torch.device("cpu"),
                    model_input=torch.ones((1, 2, 2, 2)),
                    method=ScoreCamMethod(lambda logits, target: logits[0, target].sum()),
                    objective=lambda logits, target: logits[0, target].sum(),
                    target_class=1,
                    tile_plan=plan,
                    method_params={"_selected_layer": layer},
                ))

                self.assertEqual(result.layers, {layer: expected_count})
                self.assertEqual(
                    result.available_layer_names, ("decoder 1", "decoder 2")
                )
                self.assertEqual(result.patches[0]["cam"].shape[1], 1)
                self.assertNotIn("activation", result.patches[0]["layers"][layer])

    def test_scorecam_accepts_final_class_logits_as_spatial_masks(self) -> None:
        class _OutputModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = torch.nn.Conv3d(1, 2, kernel_size=1)
                with torch.no_grad():
                    self.head.weight[:, 0, 0, 0, 0] = torch.tensor([-1.0, 1.0])
                    self.head.bias[:] = torch.tensor([0.0, 1.0])
                self.xai_layer_targets = {"decoder 1": "head"}
                self.forward_count = 0

            def forward(self, value):
                self.forward_count += 1
                return self.head(value)

        model = _OutputModel()
        image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 12
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(3, 2, 2), patch_size=2, stride=1
        )
        result = TileCollector().collect(TileCollectionRequest(
            model=model,
            device=torch.device("cpu"),
            model_input=image,
            method=ScoreCamMethod(lambda logits, target: logits[0, target].sum()),
            objective=lambda logits, target: logits[0, target].sum(),
            target_class=1,
            tile_plan=plan,
            method_params={
                "_selected_layer": "decoder 1",
                "_objective_id": "predicted_target_mask",
            },
        ))
        self.assertEqual(result.layers, {"decoder 1": 2})
        self.assertTrue(all(torch.isfinite(patch["cam"]).all() for patch in result.patches))
        self.assertTrue(any(float(patch["cam"].max()) > 0 for patch in result.patches))
        self.assertGreater(model.forward_count, 2 * len(plan.regions))

    def test_gradient_method_collects_layer_payloads_with_hooks(self) -> None:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature = torch.nn.Conv3d(1, 2, kernel_size=1)
                self.head = torch.nn.Conv3d(2, 2, kernel_size=1)
                self.xai_layer_targets = {"feature": "feature"}

            def forward(self, value):
                return self.head(self.feature(value))

        model = _Model()
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(2, 2, 2),
            patch_size=2,
            stride=0,
        )

        result = TileCollector().collect(
            TileCollectionRequest(
                model=model,
                device=torch.device("cpu"),
                model_input=torch.ones((1, 2, 2, 2), dtype=torch.float32),
                method=GradCamMethod(lambda logits, target_class: logits[0, target_class].sum()),
                objective=lambda logits, target_class: logits[0, target_class].sum(),
                target_class=1,
                tile_plan=plan,
            )
        )

        self.assertEqual(len(result.patches), len(plan.regions))
        self.assertEqual(result.layers, {"feature": 2})
        self.assertTrue(all("feature" in patch["layers"] for patch in result.patches))
        self.assertFalse(result.patches[0]["pred"].requires_grad)
        self.assertIs(result.tile_plan, plan)

    def test_gradient_method_only_collects_selected_layer(self) -> None:
        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = torch.nn.Conv3d(1, 2, kernel_size=1)
                self.second = torch.nn.Conv3d(2, 2, kernel_size=1)
                self.head = torch.nn.Conv3d(2, 2, kernel_size=1)
                self.xai_layer_targets = {"first": "first", "second": "second"}

            def forward(self, value):
                return self.head(self.second(self.first(value)))

        plan = SlidingWindowTileStrategy().plan(
            input_shape=(2, 2, 2), patch_size=2, stride=0
        )
        result = TileCollector().collect(
            TileCollectionRequest(
                model=_Model(),
                device=torch.device("cpu"),
                model_input=torch.ones((1, 2, 2, 2), dtype=torch.float32),
                method=GradCamMethod(
                    lambda logits, target_class: logits[0, target_class].sum()
                ),
                objective=lambda logits, target_class: logits[0, target_class].sum(),
                target_class=1,
                tile_plan=plan,
                method_params={"_selected_layer": "second"},
            )
        )

        self.assertEqual(result.layers, {"second": 2})
        self.assertEqual(result.available_layer_names, ("first", "second"))
        self.assertTrue(
            all(set(patch["layers"]) == {"second"} for patch in result.patches)
        )

    def test_perturbation_method_receives_runtime_params_and_reference_mask(self) -> None:
        class _Method:
            id = "perturb_test"
            family = "perturbation"
            uses_layer_controls = False

            def __init__(self) -> None:
                self.params = []
                self.requires_grad_flags = []

            def collect_patch_data(self, context):
                self.params.append(dict(context.method_params))
                self.requires_grad_flags.append(bool(context.input_tensor.requires_grad))
                return {
                    "method": self.id,
                    "pred": context.logits.detach().to("cpu"),
                    "importance_map": torch.ones_like(context.input_tensor).detach().to("cpu"),
                }

        class _Model(torch.nn.Module):
            def forward(self, value):
                mean = value.mean(dim=(2, 3, 4), keepdim=True)
                return torch.cat([mean, mean + 1.0], dim=1)

        progress = []
        method = _Method()
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(4, 4, 2),
            patch_size=2,
            stride=2,
        )

        result = TileCollector().collect(
            TileCollectionRequest(
                model=_Model(),
                device=torch.device("cpu"),
                model_input=torch.ones((1, 4, 4, 2), dtype=torch.float32),
                method=method,
                objective=lambda logits, target_class, reference_mask=None: logits[
                    0, target_class
                ].sum(),
                target_class=1,
                tile_plan=plan,
                method_params={"_progress_callback": progress.append},
                model_input_spacing=(1.0, 1.0, 2.0),
                model_input_metadata={"affine": "meta"},
                reference_mask=torch.ones((4, 4, 2), dtype=torch.bool),
                progress_units=lambda _method, _params: 5,
            )
        )

        self.assertEqual(result.layers, {"input": 1})
        self.assertEqual(progress, [{"current": 0, "total": 20}])
        self.assertEqual(method.params[0]["_tile_index"], 0)
        self.assertEqual(method.params[0]["_progress_total"], 20)
        self.assertEqual(method.params[0]["_preview_tile_origin"], (0, 0, 0))
        self.assertEqual(method.params[0]["_preview_spacing"], (1.0, 1.0, 2.0))
        self.assertEqual(method.params[0]["_preview_metadata"]["volume_id"], "perturb-preview")
        self.assertEqual(tuple(method.params[0]["reference_mask"].shape), (2, 2, 2))
        self.assertEqual(method.requires_grad_flags, [False, False, False, False])


if __name__ == "__main__":
    unittest.main()
