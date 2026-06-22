import unittest

import torch
import torch.nn.functional as F

from src.gcd.infrastructure.xai.methods.cam_methods import (
    CamPatchContext,
    GradCamMethod,
    PerturbationLimeMethod,
    SaliencyMapMethod,
    PerturbationOcclusionMethod,
    PerturbationRiseMethod,
    XResCamMethod,
)
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine


class GradCamMethodTests(unittest.TestCase):
    def test_collect_patch_data_keeps_method_pred_and_layer_tensors(self) -> None:
        layer = torch.randn((1, 2, 2, 2, 2), requires_grad=True)
        logits = layer.mean(dim=(2, 3, 4))
        method = GradCamMethod(GradCamEngine._gradcam_objective)

        payload = method.collect_patch_data(
            CamPatchContext(
                input_tensor=layer,
                logits=logits,
                layers_by_name={"layer-a": layer},
                target_class=1,
                objective=GradCamEngine._gradcam_objective,
            )
        )

        self.assertEqual(payload["method"], "gradcam")
        self.assertTrue(torch.equal(payload["pred"], logits.detach().cpu()))
        self.assertIn("layer-a", payload["layers"])
        layer_payload = payload["layers"]["layer-a"]
        self.assertEqual(layer_payload["activation"].shape, layer.shape)
        self.assertEqual(layer_payload["gradient"].shape, layer.shape)

    def test_build_tile_cam_uses_standard_gradcam_channel_weights(self) -> None:
        activation = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
        gradient = torch.linspace(0.5, 2.8, 24, dtype=torch.float32).reshape(
            1, 3, 2, 2, 2
        )
        payload = {
            "method": "gradcam",
            "layers": {
                "layer-a": {
                    "activation": activation,
                    "gradient": gradient,
                }
            },
        }
        method = GradCamMethod(GradCamEngine._gradcam_objective)

        actual = method.build_tile_cam(payload, "layer-a", 1, 3, (4, 4, 4))
        selected_activation = activation[:, 1:3, ...]
        selected_gradient = gradient[:, 1:3, ...]
        weights = torch.mean(selected_gradient, dim=(2, 3, 4), keepdim=True)
        expected = F.interpolate(
            torch.sum(selected_activation * weights, dim=1, keepdim=True),
            size=(4, 4, 4),
            mode="trilinear",
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_xrescam_build_tile_cam_matches_previous_formula(self) -> None:
        activation = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
        gradient = torch.linspace(0.5, 2.8, 24, dtype=torch.float32).reshape(
            1, 3, 2, 2, 2
        )
        payload = {
            "method": "xrescam",
            "layers": {
                "layer-a": {
                    "activation": activation,
                    "gradient": gradient,
                }
            },
        }
        method = XResCamMethod(GradCamEngine._gradcam_objective)

        actual = method.build_tile_cam(payload, "layer-a", 1, 3, (4, 4, 4))
        expected = F.interpolate(
            torch.sum(
                (activation * gradient)[:, 1:3, ...],
                dim=1,
                keepdim=True,
            ),
            size=(4, 4, 4),
            mode="trilinear",
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_saliency_map_uses_absolute_input_gradient_magnitude(self) -> None:
        input_gradient = torch.tensor(
            [
                [
                    [[[-1.0, 0.5], [2.0, -0.25]], [[0.2, -3.0], [1.5, 0.0]]],
                    [[[0.5, -2.5], [1.0, -4.0]], [[-1.0, 0.25], [2.5, -0.5]]],
                ]
            ],
            dtype=torch.float32,
        )
        payload = {
            "method": "saliency_map",
            "input_gradient": input_gradient,
        }
        method = SaliencyMapMethod(GradCamEngine._gradcam_objective)

        actual = method.build_tile_cam(payload, "ignored", 1, 3, (4, 4, 4))
        expected = F.interpolate(
            torch.amax(torch.abs(input_gradient), dim=1, keepdim=True),
            size=(4, 4, 4),
            mode="trilinear",
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_saliency_map_collect_patch_data_uses_stable_method_id(self) -> None:
        input_tensor = torch.randn((1, 1, 2, 2, 2), requires_grad=True)
        logits = torch.cat(
            [
                input_tensor.mean(dim=(2, 3, 4)),
                input_tensor.sum(dim=(2, 3, 4)),
            ],
            dim=1,
        )
        method = SaliencyMapMethod(GradCamEngine._gradcam_objective)

        payload = method.collect_patch_data(
            CamPatchContext(
                input_tensor=input_tensor,
                logits=logits,
                layers_by_name={},
                target_class=1,
                objective=GradCamEngine._gradcam_objective,
            )
        )

        self.assertEqual(payload["method"], "saliency_map")
        self.assertNotIn("layers", payload)
        self.assertEqual(payload["input_gradient"].shape, input_tensor.shape)

    def test_occlusion_masks_input_blocks_and_uses_score_drop(self) -> None:
        class _ToyModel(torch.nn.Module):
            def forward(self, value):
                return torch.cat([torch.zeros_like(value), value], dim=1)

        model = _ToyModel()
        input_tensor = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
        logits = model(input_tensor)
        method = PerturbationOcclusionMethod()
        logs = []
        progress = []
        preview_payloads = []
        full_input = torch.ones((1, 3, 3, 3), dtype=torch.float32)
        pause_controller = type(
            "PauseController",
            (),
            {"waits": 0, "wait_if_paused": lambda self: setattr(self, "waits", self.waits + 1)},
        )()

        payload = method.collect_patch_data(
            CamPatchContext(
                input_tensor=input_tensor,
                logits=logits,
                layers_by_name={},
                target_class=1,
                objective=GradCamEngine._target_logit_sum_objective,
                model=model,
                method_params={
                    "block_size": 1,
                    "stride": 1,
                    "baseline": 0.0,
                    "batch_size": 2,
                    "_score_logger": logs.append,
                    "_tile_index": 3,
                    "_progress_callback": progress.append,
                    "_progress_total": 16,
                    "_progress_offset": 8,
                    "_progress_min_interval_sec": 0,
                    "_preview_callback": preview_payloads.append,
                    "_preview_full_input": full_input,
                    "_preview_tile_origin": (1, 1, 1),
                    "_preview_permute": (0, 1, 2),
                    "_pause_controller": pause_controller,
                },
            )
        )

        self.assertEqual(payload["method"], "perturb_occlusion")
        self.assertTrue(
            torch.allclose(payload["importance_map"], torch.ones((1, 1, 2, 2, 2)))
        )
        self.assertEqual(len(logs), 8)
        self.assertIn("tile=3", logs[0])
        self.assertIn("original_score=8.000000", logs[0])
        self.assertIn("masked_score=7.000000", logs[0])
        self.assertIn("drop=1.000000", logs[0])
        self.assertEqual(progress[0], {"current": 9, "total": 16})
        self.assertEqual(progress[-1], {"current": 16, "total": 16})
        self.assertGreater(len(preview_payloads), 0)
        self.assertEqual(preview_payloads[0]["data"].shape, torch.Size((3, 3, 3)))
        self.assertEqual(preview_payloads[0]["preview_box"], ((1, 1, 1), (1, 1, 1)))
        self.assertEqual(float(preview_payloads[0]["data"][1, 1, 1]), 0.0)
        self.assertEqual(float(preview_payloads[0]["data"][0, 0, 0]), 1.0)
        self.assertGreaterEqual(pause_controller.waits, 8)

    def test_perturbation_progress_is_throttled_but_keeps_final_update(self) -> None:
        from src.gcd.infrastructure.xai.methods.perturb_occlusion import _report_progress

        progress = []
        params = {
            "_progress_callback": progress.append,
            "_progress_total": 4,
            "_progress_min_interval_sec": 60,
        }

        _report_progress(params, 1)
        _report_progress(params, 2)
        _report_progress(params, 3)
        _report_progress(params, 4)

        self.assertEqual(progress, [{"current": 1, "total": 4}, {"current": 4, "total": 4}])

    def test_perturbation_progress_default_throttle_reduces_ui_signal_pressure(self) -> None:
        from src.gcd.infrastructure.xai.methods.perturb_occlusion import _report_progress

        progress = []
        params = {"_progress_callback": progress.append, "_progress_total": 4}

        _report_progress(params, 1)
        _report_progress(params, 2)
        _report_progress(params, 3)

        self.assertEqual(progress, [{"current": 1, "total": 4}])

    def test_lime_returns_reproducible_input_importance_map(self) -> None:
        class _ToyModel(torch.nn.Module):
            def forward(self, value):
                return torch.cat([torch.zeros_like(value), value], dim=1)

        model = _ToyModel()
        input_tensor = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
        logits = model(input_tensor)
        method = PerturbationLimeMethod()
        context = CamPatchContext(
            input_tensor=input_tensor,
            logits=logits,
            layers_by_name={},
            target_class=1,
            objective=GradCamEngine._target_logit_sum_objective,
            model=model,
            method_params={
                "segments_per_axis": 2,
                "num_samples": 16,
                "kernel_width": 1.0,
                "baseline": 0.0,
                "batch_size": 4,
                "random_seed": 7,
            },
        )

        first = method.collect_patch_data(context)["importance_map"]
        second = method.collect_patch_data(context)["importance_map"]

        self.assertEqual(first.shape, (1, 1, 2, 2, 2))
        self.assertTrue(torch.allclose(first, second))
        self.assertFalse(torch.isnan(first).any())

    def test_rise_returns_reproducible_input_importance_map(self) -> None:
        class _ToyModel(torch.nn.Module):
            def forward(self, value):
                return torch.cat([torch.zeros_like(value), value], dim=1)

        model = _ToyModel()
        input_tensor = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
        logits = model(input_tensor)
        method = PerturbationRiseMethod()
        context = CamPatchContext(
            input_tensor=input_tensor,
            logits=logits,
            layers_by_name={},
            target_class=1,
            objective=GradCamEngine._target_logit_sum_objective,
            model=model,
            method_params={
                "num_masks": 8,
                "mask_grid_size": 2,
                "keep_probability": 0.5,
                "baseline": 0.0,
                "batch_size": 4,
                "random_seed": 11,
            },
        )

        first = method.collect_patch_data(context)["importance_map"]
        second = method.collect_patch_data(context)["importance_map"]

        self.assertEqual(first.shape, (1, 1, 2, 2, 2))
        self.assertTrue(torch.allclose(first, second))
        self.assertFalse(torch.isnan(first).any())


if __name__ == "__main__":
    unittest.main()
