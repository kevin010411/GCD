import unittest

import torch

from src.gcd.infrastructure.cam_methods import GradCamMethod
from src.gcd.infrastructure.core_engine import GradCamEngine


class CoreEngineCamMethodTests(unittest.TestCase):
    def test_unknown_method_falls_back_to_gradcam(self) -> None:
        logs = []
        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = logs.append
        engine.cam_methods = {"gradcam": GradCamMethod(GradCamEngine._gradcam_objective)}
        engine.active_method_id = "gradcam"

        method = engine._resolve_cam_method("mystery")

        self.assertEqual(method.id, "gradcam")
        self.assertTrue(any("未知 CAM method" in message for message in logs))

    def test_compute_cam_keeps_gradcam_output_normalized(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.cam_methods = {"gradcam": GradCamMethod(GradCamEngine._gradcam_objective)}
        engine.active_method_id = "gradcam"
        engine.file_name = "sample.nii.gz"
        engine.cfg = {"default_layer": "layer-a"}
        engine.layers = {"layer-a": 1}
        engine.SIZE = 2
        engine.STRIDE = 0
        engine.PERMUTE = (0, 1, 2)
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.save_dir = None
        pred = torch.tensor([[[[[0.1]]], [[[0.9]]]]], dtype=torch.float32)
        tile_payload = {
            "method": "gradcam",
            "pred": pred,
            "layers": {
                "layer-a": {
                    "activation": torch.ones((1, 1, 1, 1, 1), dtype=torch.float32),
                    "gradient": torch.full(
                        (1, 1, 1, 1, 1), 2.0, dtype=torch.float32
                    ),
                }
            },
        }
        engine.patch = [tile_payload, tile_payload, tile_payload, tile_payload]

        selected = engine.compute_cam(method="gradcam")

        self.assertEqual(selected, "layer-a")
        self.assertGreaterEqual(float(engine.cam.min()), 0.0)
        self.assertLessEqual(float(engine.cam.max()), 1.0)

    def test_dataset_input_omits_large_result_and_patch_payloads(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine.cam = torch.ones((2, 2, 2), dtype=torch.float32)
        engine.volume_data = torch.ones((2, 2, 2), dtype=torch.float32)
        engine.img0 = None
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.origin_img = None
        engine.origin_meta = {}
        engine.origin_shape = (2, 2, 2)
        engine.img1_spacing = (1.0, 1.0, 1.0)
        engine.display_metadata = {}
        engine.layers = {"layer-a": 1}
        engine.file_name = "sample.nii.gz"
        engine.patch = [{"method": "gradcam", "layers": {"layer-a": torch.ones(1)}}]
        engine.target_class = 1
        engine.active_method_id = "gradcam"
        engine.model_output = torch.zeros((2, 2, 2), dtype=torch.float32)
        engine.xai_cache_key = "cfg.py|1|gradcam"

        state = engine.dataset_input()

        self.assertFalse(hasattr(state, "cam"))
        self.assertFalse(hasattr(state, "volume_data"))
        self.assertFalse(hasattr(state, "patch"))
        self.assertFalse(hasattr(state, "model_output"))
        self.assertIsNotNone(state.img1)


if __name__ == "__main__":
    unittest.main()
