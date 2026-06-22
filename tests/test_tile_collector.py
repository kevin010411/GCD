import unittest

import torch

from src.gcd.infrastructure.xai.methods.cam_methods import GradCamMethod
from src.gcd.infrastructure.xai.tiling.tile_collector import (
    TileCollectionRequest,
    TileCollector,
)
from src.gcd.infrastructure.xai.tiling.tile_strategy import LegacyFourTileStrategy


class TileCollectorTests(unittest.TestCase):
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
        plan = LegacyFourTileStrategy().plan(
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

        self.assertEqual(len(result.patches), 4)
        self.assertEqual(result.layers, {"feature": 2})
        self.assertTrue(all("feature" in patch["layers"] for patch in result.patches))
        self.assertIs(result.tile_plan, plan)

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
        plan = LegacyFourTileStrategy().plan(
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
