"""Exercise label-free CAM export and resume through the public runner."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from experiments.src.benchmark_runner import parse_args, run


class HeatmapOnlyTests(unittest.TestCase):
    def test_export_without_label_or_metrics_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, config, checkpoint = root / "image.nii.gz", root / "model.py", root / "model.pth"
            affine = np.diag([1., 1., 1., 1.])
            affine[:3, 3] = [10., -20., 30.]
            data = np.arange(64, dtype=np.float32).reshape(4, 4, 4) / 63
            nib.save(nib.Nifti1Image(data, affine), source)
            checkpoint.write_bytes(b"mock runtime checkpoint")
            config.write_text(
                f"ckpt = {str(checkpoint)!r}\n"
                "model = dict(type='Toy')\n"
                "preprocessing = dict(channel_dim=None, force_3d=False, spacing=(1.,1.,1.), "
                "steps=[dict(type='EnsureType', dtype='float32')])\n"
                "inference = dict(device='cpu', roi_size=(4,4,4), sw_batch_size=1, "
                "overlap=0.25, blend_mode='gaussian')\n", encoding="utf-8")
            model = torch.nn.Sequential(torch.nn.Conv3d(1, 1, 1, bias=False),
                                        torch.nn.Conv3d(1, 1, 1, bias=False),
                                        torch.nn.Conv3d(1, 2, 1, bias=True))
            for layer in model:
                layer.weight.data.fill_(1.)
            model[-1].weight.data[0].zero_()
            model[-1].bias.data.copy_(torch.tensor([0., 1.]))
            def logits(batch):
                with torch.no_grad():
                    return model(batch)
            predictor = SimpleNamespace(runtime=SimpleNamespace(model=model), logits=logits)
            args = parse_args([
                "--model-config", str(config), "--case", "toy", str(source),
                "--heatmap-only", "--export-nifti", "--layers", "0", "1", "2",
                "--methods", "gradcam_L1", "--output", str(root / "output"),
            ])
            with patch("src.gcd.infrastructure.xai.evaluation.ConfiguredPredictor", return_value=predictor), \
                 patch("experiments.src.benchmark_runner.ConfiguredPredictor", return_value=predictor), \
                 patch("experiments.src.benchmark_runner.evaluate_heatmap", side_effect=AssertionError("GT evaluation called")):
                output = run(args)
                exported = nib.load(output / "toy/gradcam_L1/heatmap_original_space.nii.gz")
                actual = np.asarray(exported.dataobj)
                self.assertEqual(actual.shape, data.shape)
                np.testing.assert_allclose(exported.affine, affine)
                self.assertTrue(np.isfinite(actual).all())
                self.assertGreater(float(actual.max()), 0.)
                self.assertGreaterEqual(float(actual.min()), 0.)
                self.assertLessEqual(float(actual.max()), 1.)
                self.assertFalse((output / "toy/gradcam_L1/curve.csv").exists())
                prediction_path = output / "toy/prediction_original_space.nii.gz"
                predicted = nib.load(prediction_path)
                np.testing.assert_array_equal(np.asarray(predicted.dataobj), np.ones_like(data))
                np.testing.assert_allclose(predicted.affine, affine)
                self.assertEqual(predicted.get_data_dtype(), np.dtype('uint8'))
                np.testing.assert_array_equal(np.load(output / "toy/prediction_model_space.npy"),
                                              np.ones_like(data))
                with patch("experiments.src.benchmark_runner.generate", side_effect=AssertionError("cached CAM recomputed")):
                    self.assertEqual(run(args), output)

    def test_evaluation_still_requires_label(self):
        args = parse_args(["--model-config", "unused.py", "--case", "toy", "image.nii.gz",
                           "--output", "unused"])
        with self.assertRaisesRegex(ValueError, "ID IMAGE LABEL"):
            run(args)
