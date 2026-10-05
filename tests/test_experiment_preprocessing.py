"""Config inheritance and exact ACDC preprocessing/model parity."""
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from mmengine import Config
from monai.data import MetaTensor
from monai.transforms import Compose, Spacing, Orientation, SpatialPad, NormalizeIntensity, EnsureType

from experiments.src.preprocessing import preprocess_tensor, preprocess_array, load_image

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingTests(unittest.TestCase):
    def test_all_shared_configs_switch_without_loading_weights(self):
        from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
        engine = GradCamEngine(str(ROOT / "config/model/unet_3d.py"))
        for path in (ROOT / "config/model").glob("*.py"):
            engine.set_config(str(path))
            self.assertIn(engine.cfg.default_layer, engine.model_layer_metadata()["layer_names"])
            self.assertTrue(engine.cfg.preprocessing.steps)
            self.assertEqual(len(engine.cfg.inference.roi_size), 3)

    def test_model_configs_have_no_legacy_fields(self):
        from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
        for path in (ROOT / "config/model").glob("*.py"):
            cfg = Config.fromfile(str(path))
            for obsolete in ("size", "stride", "spacing", "permute", "xai_tiling"):
                self.assertNotIn(obsolete, cfg)
            self.assertEqual(cfg.inference.tile_strategy, "sliding_window")
            self.assertIn("permute", cfg.display)
            self.assertTrue(cfg.preprocessing.steps)
            engine = GradCamEngine(str(path))
            self.assertEqual(engine.SPACING, cfg.preprocessing.spacing)
            self.assertEqual(engine.PERMUTE, cfg.display.permute)
        self.assertFalse((ROOT / "config/base.py").exists())

    def test_gcd_padding_uses_rectangular_roi_and_rejects_legacy(self):
        cfg = Config.fromfile(str(ROOT / "config/model/unet_3d.py"))
        cfg.inference.roi_size = (8, 6, 4)
        cfg.preprocessing.steps = [dict(type="SpatialPad", spatial_size="window", mode="constant")]
        image = MetaTensor(torch.ones(1, 3, 3, 2))
        self.assertEqual(tuple(preprocess_tensor(image, cfg).shape), (1, 8, 6, 4))
        cfg.preprocessing.steps[0].spatial_size = "legacy_window"
        with self.assertRaisesRegex(ValueError, "legacy_window"):
            preprocess_tensor(image, cfg)

    def test_gcd_configs_inherit_window_and_transforms(self):
        for name in ("predict", "scorecam_unet3d_audit"):
            cfg = Config.fromfile(str(ROOT / "experiments/src/configs" / f"{name}.py"))
            self.assertEqual(cfg.inference.roi_size, (128, 128, 128))
            self.assertEqual(cfg.inference.blend_mode, "constant")
            self.assertEqual(cfg.preprocessing.steps[0].type, "Spacing")

    def test_acdc_order_labels_and_array_match(self):
        cfg = Config.fromfile(str(ROOT / "experiments/src/configs/xai_hw_acdc.py"))
        cfg.inference.roi_size = (8, 8, 4)
        affine = np.diag([-1.5, -1.5, 5., 1.])
        array = np.arange(6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3)
        image = MetaTensor(torch.tensor(array)[None], affine=torch.tensor(affine))
        reference = Compose([
            Spacing((1.5, 1.5, 5.), mode="bilinear"), Orientation(axcodes="RAS"),
            SpatialPad((8, 8, 4), mode="constant", method="symmetric"),
            NormalizeIntensity(nonzero=True, channel_wise=True), EnsureType(dtype=torch.float32),
        ])(image.clone())
        actual = preprocess_array(array, affine, cfg)
        torch.testing.assert_close(actual.as_tensor(), reference.as_tensor(), rtol=0, atol=0)
        torch.testing.assert_close(actual.affine, reference.affine, rtol=0, atol=0)
        label = preprocess_tensor((image > 40).to(torch.float32), cfg, label=True)
        self.assertEqual(set(label.unique().tolist()), {0., 1.})
        self.assertEqual(tuple(label.shape), (1, 8, 8, 4))

    def test_explicit_small_padding_is_completed_to_cam_roi_after_transforms(self):
        cfg = Config.fromfile(str(ROOT / "config/model/xai_hw_acdc.py"))
        cfg.inference.roi_size = (8, 8, 4)
        cfg.preprocessing.steps = [
            dict(type="SpatialPad", spatial_size=(4, 4, 2), mode="constant"),
            dict(type="NormalizeIntensity", nonzero=True, channel_wise=True),
            dict(type="EnsureType", dtype="float32"),
        ]
        image = MetaTensor(torch.arange(32, dtype=torch.float32).reshape(1, 4, 4, 2))
        expected = Compose([
            SpatialPad((4, 4, 2), mode="constant"),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            EnsureType(dtype=torch.float32), SpatialPad((8, 8, 4), method="symmetric", mode="constant"),
        ])(image.clone())
        actual = preprocess_tensor(image, cfg)
        torch.testing.assert_close(actual.as_tensor(), expected.as_tensor(), rtol=0, atol=0)
        torch.testing.assert_close(actual.affine, expected.affine, rtol=0, atol=0)

    def test_config_model_loads_real_checkpoint_and_patient_matches_xai_hw(self):
        checkpoint = Path("/workspace/xai_hw/assignment/colab/checkpoint/best_model.pth")
        if not checkpoint.is_file():
            self.skipTest("real xai_hw data is not present")
        cfg = Config.fromfile(str(ROOT / "experiments/src/configs/xai_hw_acdc.py"))
        import src.model
        from src.utils import build_model
        model = build_model(cfg.model)
        payload = torch.load(cfg.ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
        sys.path.insert(0, "/workspace/xai_hw/assignment/colab")
        from benchmark_suite import reference
        path = Path("/workspace/xai_hw/assignment/colab/data/testing/patient103/patient103_frame01.nii.gz")
        actual = load_image(path, cfg)
        from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
        from src.gcd.infrastructure.model_input_preprocessor import ModelInputPreprocessor
        import monai.transforms as mt
        engine = GradCamEngine(str(ROOT / "config/model/xai_hw_acdc.py"))
        raw = mt.EnsureChannelFirst(channel_dim="no_channel")(
            mt.LoadImage(image_only=True)(str(path))
        )
        ui = ModelInputPreprocessor().preprocess(
            img0=raw, origin_img=None, origin_meta=raw.meta, origin_shape=raw.shape,
            config=engine._model_input_preprocess_config(),
        )
        torch.testing.assert_close(ui.img1.as_tensor(), actual.as_tensor(), rtol=0, atol=0)
        torch.testing.assert_close(ui.img1.affine, actual.affine, rtol=0, atol=0)
        # MONAI 1.2 cannot accept the reference's tuple dtype in EnsureTyped.
        # Use the identical scalar float32 cast, keeping every spatial/intensity step.
        from monai.transforms import EnsureTyped
        reference_steps = reference.preprocess().transforms[:-1]
        expected = Compose([*reference_steps, EnsureTyped(keys="image", dtype=torch.float32)])(
            {"image": str(path)}
        )["image"]
        torch.testing.assert_close(actual.as_tensor(), expected.as_tensor(), rtol=0, atol=0)
        torch.testing.assert_close(actual.affine, expected.affine, rtol=0, atol=0)
        torch.manual_seed(123)
        model.eval()
        ref_model = reference.build_model("cpu")
        with torch.inference_mode():
            tile = torch.randn(1, 1, 16, 16, 8)
            torch.testing.assert_close(model(tile), ref_model(tile), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
