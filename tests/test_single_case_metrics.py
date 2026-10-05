import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from mmengine import Config
from monai.data import MetaTensor

from experiments.src.single_case_runner import parse_args, run
from src.gcd.infrastructure.xai.evaluation import load_heatmap, load_case
from src.gcd.infrastructure.xai.benchmark_metrics import evaluate_heatmap, behaviour_curve, auc


ROOT = Path(__file__).resolve().parents[1]


class SingleCaseMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = np.ones((4, 4, 4), np.float32)
        self.label = np.zeros_like(self.image)
        self.label[1:3, 1:3, 1:3] = 1
        self.heatmap = self.label.copy()
        for name, array in (("image", self.image), ("label", self.label), ("heatmap", self.heatmap)):
            nib.save(nib.Nifti1Image(array, np.eye(4)), self.root / f"{name}.nii.gz")
        self.cfg = Config(dict(
            preprocessing=dict(channel_dim="no_channel", spacing=(1, 1, 1),
                               steps=[dict(type="EnsureType", dtype="float32")]),
            inference=dict(roi_size=(4, 4, 4), device="cpu", overlap=.25,
                           sw_batch_size=1, blend_mode="constant", tile_strategy="sliding_window"),
            postprocessing=dict(keep_largest_connected_component=False),
            model=dict(type="UNet", out_channels=2), ckpt=str(self.root / "checkpoint.pth"),
        ))
        self.cfg.dump(str(self.root / "model.py"))
        (self.root / "checkpoint.pth").write_bytes(b"test fixture")

    def test_original_grid_checks_affine_even_when_shape_matches(self):
        meta = MetaTensor(torch.ones(1, 4, 4, 4), affine=torch.eye(4))
        actual = load_heatmap(self.root / "heatmap.nii.gz", self.root / "image.nii.gz", meta)
        np.testing.assert_array_equal(actual, self.heatmap)
        affine = np.eye(4)
        affine[0, 3] = 10
        nib.save(nib.Nifti1Image(self.heatmap, affine), self.root / "wrong.nii.gz")
        with self.assertRaisesRegex(ValueError, "shape/affine"):
            load_heatmap(self.root / "wrong.nii.gz", self.root / "image.nii.gz", meta)
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            load_heatmap(self.root / "wrong.nii.gz", self.root / "image.nii.gz", meta, "world")

    def test_resampling_does_not_apply_ct_intensity_transform(self):
        meta = MetaTensor(torch.ones(1, 4, 4, 4), affine=torch.eye(4))
        values = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        nib.save(nib.Nifti1Image(values, np.eye(4)), self.root / "heatmap.nii.gz")
        actual = load_heatmap(self.root / "heatmap.nii.gz", self.root / "image.nii.gz", meta)
        np.testing.assert_allclose(actual, values / 63, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "shape/affine"):
            load_heatmap(self.root / "heatmap.nii.gz", self.root / "image.nii.gz",
                         MetaTensor(torch.ones(1, 8, 4, 4), affine=torch.eye(4)), "model")

    def test_label_geometry_and_values_are_validated(self):
        affine = np.eye(4)
        affine[2, 3] = 2
        nib.save(nib.Nifti1Image(self.label, affine), self.root / "label.nii.gz")
        with self.assertRaisesRegex(ValueError, "Image/label"):
            load_case(self.root / "image.nii.gz", self.root / "label.nii.gz", self.cfg)
        nib.save(nib.Nifti1Image(self.label + .1, np.eye(4)), self.root / "label.nii.gz")
        with self.assertRaisesRegex(ValueError, "integer"):
            load_case(self.root / "image.nii.gz", self.root / "label.nii.gz", self.cfg)

    def test_shared_evaluator_matches_existing_curve_and_auc(self):
        infer = lambda x: ((x > .5).astype(np.int64), x)
        kwargs = dict(image=self.image, saliency=self.heatmap, label=self.label,
                      class_id=1, full_prediction=np.ones_like(self.label),
                      full_probability=self.image, infer=infer, steps=3)
        old_rows, _ = behaviour_curve(**kwargs)
        rows, summary = evaluate_heatmap(**kwargs, spacing=(1, 1, 1), preserve_answer=True)
        self.assertEqual([dict((k, v) for k, v in r.items() if k != "variant")
                          for r in rows if r["variant"] == "standard"], old_rows)
        self.assertEqual(summary["standard_insertion_gt_dice_auc"],
                         auc(old_rows, "gt_dice", "insertion"))
        self.assertEqual(len(rows), 12)

    def test_no_prediction_keeps_absolute_dice_and_marks_ratios_undefined(self):
        rows, summary = evaluate_heatmap(
            image=self.image, saliency=self.heatmap, label=self.label, class_id=1,
            full_prediction=np.zeros_like(self.label), full_probability=self.image,
            infer=lambda x: (np.zeros_like(x), x), spacing=(1, 1, 1), steps=2,
        )
        self.assertEqual(rows[0]["gt_dice"], 0)
        self.assertTrue(np.isnan(rows[0]["relative_gt_dice"]))
        self.assertFalse(summary["relative_dice_defined"])
        self.assertFalse(summary["fixed_roi_probability_defined"])

    def test_cli_writes_curves_summary_plots_and_protocol_without_recomputing_cam(self):
        class Predictor:
            runtime = type("Runtime", (), {"device": "cpu"})()
            def __init__(self, cfg):
                pass
            def __call__(self, volume, class_id):
                return (volume > .5).astype(np.int64), volume
        argv = [
            "--image", str(self.root / "image.nii.gz"),
            "--label", str(self.root / "label.nii.gz"),
            "--heatmap", str(self.root / "heatmap.nii.gz"),
            "--model-config", str(self.root / "model.py"),
            "--steps", "3", "--preserve-answer", "--output", str(self.root / "output"),
        ]
        with patch("experiments.src.single_case_runner.ConfiguredPredictor", Predictor):
            output = run(parse_args(argv))
            with self.assertRaises(FileExistsError):
                run(parse_args(argv))
        summary = json.loads((output / "summary.json").read_text())
        self.assertIn("standard_deletion_gt_dice_auc", summary)
        self.assertIn("preserve_answer_insertion_relative_gt_dice_auc", summary)
        self.assertEqual(len(list((output / "plots").glob("*.png"))), 6)
        self.assertEqual(len((output / "curve.csv").read_text().splitlines()), 13)
        self.assertTrue((output / "heatmap_model_space.nii.gz").is_file())
        self.assertEqual(json.loads((output / "protocol.json").read_text())["actual_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
