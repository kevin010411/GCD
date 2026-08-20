import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from experiments.xai import (
    insertion_deletion_metrics,
    normalize_attribution,
    target_class_from_prediction,
)
from experiments.outputs import export_csv_files, export_faithfulness_plots


class ExperimentsXaiTests(unittest.TestCase):
    def test_normalization_handles_constant_map(self):
        self.assertTrue(torch.equal(normalize_attribution(torch.ones(2, 2, 2)), torch.zeros(2, 2, 2)))

    def test_auto_target_selects_largest_foreground_class(self):
        prediction = torch.tensor([[[0, 1], [2, 2]]])
        self.assertEqual(target_class_from_prediction(prediction, "auto"), 2)

    def test_insertion_and_deletion_return_curves_and_auc(self):
        batch = torch.tensor([[[[[1.0, 0.8], [0.2, 0.0]]]]])
        attribution = batch[0, 0].clone()
        mask = torch.ones_like(attribution, dtype=torch.bool)
        cfg = SimpleNamespace(metrics=SimpleNamespace(perturbation_steps=2, perturbation_baseline=0.0))

        def infer(value):
            foreground = value[:, 0]
            return torch.stack((torch.zeros_like(foreground), foreground), dim=1)

        result = insertion_deletion_metrics(
            batch, attribution, infer, 1, mask, cfg, progress=False
        )
        self.assertEqual(result["fractions"], [0.0, 0.5, 1.0])
        self.assertGreater(result["insertion_auc"], 0.0)
        self.assertGreater(result["deletion_scores"][0], result["deletion_scores"][-1])

    def test_exports_csv_tables_and_faithfulness_plot(self):
        metrics = {
            "input": "input.nii.gz", "output": "prediction.nii.gz",
            "config": "config.py", "checkpoint": "model.pth", "device": "cpu",
            "class_count": 2, "class_voxels": {"0": 3, "1": 1},
            "benchmark": {
                "latency_seconds": [1.0], "mean_latency_seconds": 1.0,
                "volumes_per_second": 1.0, "parameter_count": 10,
            },
            "xai": {"saliency_map": {
                "target_class": 1, "attribution": "xai.nii.gz",
                "fractions": [0.0, 1.0], "insertion_scores": [0.1, 0.9],
                "deletion_scores": [0.9, 0.1], "insertion_auc": 0.5,
                "deletion_auc": 0.5,
            }},
        }
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            csv_paths = export_csv_files(metrics, output_dir)
            plot_paths = export_faithfulness_plots(metrics, output_dir)
            self.assertTrue(all(path.is_file() for path in csv_paths + plot_paths))
            self.assertIn("insertion_score", (output_dir / "xai_curves.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
