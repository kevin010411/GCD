"""Protocol checks for the tiled, model-space XAI benchmark."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.src.xai.benchmark_cam import feature_layers, patch_cam, window_slices
from experiments.src.xai.benchmark_metrics import auc, behaviour_curve, fractions, overlap
from experiments.src.xai_design import BenchmarkTiledCamMethod, XaiExecutionContext


class BenchmarkProtocolTests(unittest.TestCase):
    def test_dense21_covers_small_structures_and_full_interval(self):
        points = fractions(21)
        self.assertEqual(len(points), 21)
        self.assertEqual(float(points[0]), 0.0)
        self.assertAlmostEqual(float(points[1]), .0005)
        self.assertEqual(float(points[-1]), 1.0)
        self.assertEqual(overlap(np.zeros(2), np.zeros(2)), (0.0, 0.0))

    def test_tiled_windows_cover_boundary_voxels(self):
        shape, roi = (5, 4, 3), (3, 3, 3)
        coverage = np.zeros(shape, np.int32)
        for spatial in window_slices(shape, roi, .25):
            coverage[spatial] += 1
        self.assertTrue(np.all(coverage > 0))
        self.assertEqual(int(coverage[0, 0, 0]), 1)
        self.assertGreaterEqual(int(coverage[-1, -1, -1]), 1)

    def test_auto_layers_exclude_final_logit_convolution(self):
        model = torch.nn.Sequential(*(torch.nn.Conv3d(1, 1, 1) for _ in range(4)))
        layers = feature_layers(model)
        self.assertEqual([name for name, _ in layers], ["0", "1", "2"])

    def test_decoder_input_can_be_used_as_feature_map(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature = torch.nn.Conv3d(1, 2, 1, bias=False)
                self.output = torch.nn.Conv3d(2, 2, 1, bias=False)
                self.xai_layer_targets = {"decoder 1": "output"}
                with torch.no_grad():
                    self.feature.weight.fill_(1)
                    self.output.weight.fill_(1)

            def forward(self, value):
                return self.output(self.feature(value).relu())

        model = Toy()
        _, tap = feature_layers(model, (
            "decoder 1:input", "feature", "output",
        ))[0]
        image = torch.ones((1, 1, 2, 2, 2))
        cam = patch_cam(model, tap, image, 1, torch.ones((2, 2, 2), dtype=torch.bool),
                        "hirescam")
        self.assertEqual(tuple(cam.shape), (2, 2, 2))
        self.assertTrue(torch.isfinite(cam).all())
        self.assertGreater(float(cam.sum()), 0.0)

    def test_curve_uses_stable_ties_and_full_input_relative_dice(self):
        image = np.ones((2, 2, 2), np.float32)
        heatmap = np.zeros_like(image)
        label = np.zeros_like(image)
        label.flat[0] = 1
        full_prediction = np.ones_like(image, np.int64)
        full_probability = np.ones_like(image)

        def infer(volume):
            prediction = (volume > .5).astype(np.int64)
            return prediction, volume.astype(np.float32)

        curve, full = behaviour_curve(
            image=image, saliency=heatmap, label=label, class_id=1,
            full_prediction=full_prediction, full_probability=full_probability,
            infer=infer, steps=3, modes=("insertion", "deletion"),
        )
        inserted = [row for row in curve if row["mode"] == "insertion"]
        deleted = [row for row in curve if row["mode"] == "deletion"]
        self.assertAlmostEqual(full["full_gt_dice"], 2 / 9)
        self.assertEqual(inserted[0]["relative_gt_dice"], 0.0)
        self.assertAlmostEqual(inserted[1]["gt_dice"], 2 / 5)
        self.assertAlmostEqual(inserted[-1]["relative_gt_dice"], 1.0)
        self.assertEqual(deleted[1]["gt_dice"], 0.0)
        self.assertAlmostEqual(auc(curve, "relative_gt_dice", "insertion"), 1.15)

    def test_preserved_answer_has_separate_curve(self):
        image = np.ones((2, 2, 2), np.float32)
        label = np.zeros_like(image)
        label.flat[0] = 1

        def infer(volume):
            return (volume > .5).astype(np.int64), volume.astype(np.float32)

        curve, _ = behaviour_curve(
            image=image, saliency=np.zeros_like(image), label=label, class_id=1,
            full_prediction=np.ones_like(image, np.int64),
            full_probability=np.ones_like(image), infer=infer,
            steps=2, preserve_answer=True, modes=("insertion", "deletion"),
        )
        self.assertGreater(curve[0]["gt_dice"], 0.0)
        self.assertGreater(curve[-1]["gt_dice"], 0.0)

    def test_existing_predict_can_request_benchmark_heatmap(self):
        logits = torch.zeros((1, 2, 2, 2, 2))
        logits[0, 1] = torch.arange(8).reshape(2, 2, 2)
        cfg = SimpleNamespace(
            inference=SimpleNamespace(roi_size=(2, 2, 2), overlap=.25),
            preprocessing=SimpleNamespace(spacing=(1., 1., 1.)),
        )
        context = XaiExecutionContext(
            batch=torch.ones((1, 1, 2, 2, 2)), model=torch.nn.Identity(),
            predictor=None, target_class=1, target_mask=torch.ones((2, 2, 2)),
            cfg=cfg, dataset_context={"baseline_logits": logits},
        )
        method = BenchmarkTiledCamMethod(id="logit", method="prediction_logits_minmax")
        heatmap, metadata = method.explain(context)
        self.assertEqual(tuple(heatmap.shape), (2, 2, 2))
        self.assertAlmostEqual(float(heatmap.min()), 0.0)
        self.assertAlmostEqual(float(heatmap.max()), 1.0)
        self.assertEqual(metadata["pipeline"], "benchmark_tiled")


if __name__ == "__main__":
    unittest.main()
