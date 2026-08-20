import unittest
from types import SimpleNamespace

import torch

from experiments.xai import (
    insertion_deletion_metrics,
    normalize_attribution,
    target_class_from_prediction,
)


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

        result = insertion_deletion_metrics(batch, attribution, infer, 1, mask, cfg)
        self.assertEqual(result["fractions"], [0.0, 0.5, 1.0])
        self.assertGreater(result["insertion_auc"], 0.0)
        self.assertGreater(result["deletion_scores"][0], result["deletion_scores"][-1])


if __name__ == "__main__":
    unittest.main()
