"""Layer discovery tests without a Qt runtime or checkpoint dependency."""
import unittest
from pathlib import Path
from unittest.mock import patch

from mmengine import Config
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine

ROOT = Path(__file__).resolve().parents[1]


class ModelLayerDiscoveryTests(unittest.TestCase):
    def test_configs_do_not_require_duplicate_layer_lists(self):
        for path in (ROOT / "config/model").glob("*.py"):
            cfg = Config.fromfile(str(path))
            self.assertNotIn("xai_layers", cfg, str(path))

    def test_actual_small_models_discover_layers_and_output_channels(self):
        for path, overrides, roi in (
            ("unet_3d.py", {"channels": (4, 8, 16, 16)}, (16, 16, 16)),
            ("nn_unetr.py", {"filters": [4, 8, 16, 32, 64]}, (32, 32, 32)),
            ("xai_hw_acdc.py", {"channels": (4, 8, 16, 16)}, (16, 16, 16)),
        ):
            with self.subTest(path=path):
                engine = GradCamEngine(str(ROOT / "config/model" / path))
                engine.cfg.model.update(overrides)
                engine.cfg.inference.roi_size = roi
                with patch("torch.cuda.is_available", return_value=False):
                    metadata = engine.model_layer_metadata()
                self.assertIn(engine.cfg.default_layer, metadata["layer_names"])
                self.assertEqual(set(metadata["feature_sizes"]), set(metadata["layer_names"]))
                self.assertTrue(all(count > 0 for count in metadata["feature_sizes"].values()))
                self.assertGreater(metadata["feature_size"], 0)
                # Actual model/ROI changes invalidate cached scalar metadata.
                engine.cfg.model.out_channels += 1
                with patch("torch.cuda.is_available", return_value=False):
                    changed = engine.model_layer_metadata()
                self.assertNotEqual(metadata["model_key"], changed["model_key"])


if __name__ == "__main__":
    unittest.main()
