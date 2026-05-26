import unittest

import torch
import torch.nn.functional as F

from src.gcd.infrastructure.cam_methods import (
    CamPatchContext,
    GradCamMethod,
    SaliencyMapMethod,
    XResCamMethod,
)
from src.gcd.infrastructure.core_engine import GradCamEngine


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


if __name__ == "__main__":
    unittest.main()
