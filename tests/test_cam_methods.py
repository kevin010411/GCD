import unittest

import torch
import torch.nn.functional as F

from src.gcd.infrastructure.cam_methods import GradCamMethod
from src.gcd.infrastructure.core_engine import GradCamEngine


class GradCamMethodTests(unittest.TestCase):
    def test_collect_patch_data_keeps_method_pred_and_layer_tensors(self) -> None:
        layer = torch.randn((1, 2, 2, 2, 2), requires_grad=True)
        logits = layer.mean(dim=(2, 3, 4))
        method = GradCamMethod(GradCamEngine._gradcam_objective)

        payload = method.collect_patch_data({"layer-a": layer}, logits, 1)

        self.assertEqual(payload["method"], "gradcam")
        self.assertTrue(torch.equal(payload["pred"], logits.detach().cpu()))
        self.assertIn("layer-a", payload["layers"])
        layer_payload = payload["layers"]["layer-a"]
        self.assertEqual(layer_payload["activation"].shape, layer.shape)
        self.assertEqual(layer_payload["gradient"].shape, layer.shape)

    def test_build_tile_cam_matches_previous_gradcam_formula(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
