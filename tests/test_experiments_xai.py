import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import numpy as np

from experiments.xai import (
    compute_organ_occlusion_attribution,
    insertion_deletion_metrics,
    normalize_attribution,
    normalize_signed_attribution,
    target_class_from_prediction,
)
from experiments.outputs import export_csv_files, export_faithfulness_plots
from experiments.xai_design import (
    FaithfulnessAnswerAggregator,
    OrganOcclusionXaiMethod,
    PredictionDiceScore,
    build_xai_answer,
    build_xai_methods,
    build_xai_metrics,
    OrganOccluder,
)
from experiments.runner import _xai_execution_enabled
from src.gcd.domain import OrganMaskRecord


class ExperimentsXaiTests(unittest.TestCase):
    def test_mmengine_builds_xai_strategy_graph(self):
        cfg = {
            "XaiMethods": [dict(
                type="OrganOcclusionXaiMethod",
                id="organ",
                objective=dict(type="PredictionDiceScore"),
            )],
            "XaiMetrics": [dict(
                type="PerturbationInsertion",
                methods=["organ"], target_class=[1], steps=2,
                scorers=[dict(type="PredictionDiceScore")],
            )],
            "XaiAnswer": dict(type="FaithfulnessAnswerAggregator"),
        }
        methods = build_xai_methods(cfg)
        metrics = build_xai_metrics(cfg)
        answer = build_xai_answer(cfg)
        self.assertIsInstance(methods["organ"], OrganOcclusionXaiMethod)
        self.assertEqual(methods["organ"].target_classes, ("auto",))
        self.assertIsInstance(methods["organ"].objective, PredictionDiceScore)
        self.assertIsInstance(metrics[0].scorers[0], PredictionDiceScore)
        self.assertIsInstance(answer, FaithfulnessAnswerAggregator)

    def test_xai_method_accepts_standalone_target_classes(self):
        methods = build_xai_methods({"XaiMethods": [dict(
            type="OrganOcclusionXaiMethod",
            id="organ",
            target_class=[1, 2, 3],
        )]})
        self.assertEqual(methods["organ"].target_classes, (1, 2, 3))

    def test_metric_builder_assigns_stable_operation_indexes(self):
        metrics = build_xai_metrics({"XaiMetrics": [
            dict(type="PerturbationInsertion", methods=["x"], target_class=[1]),
            dict(type="PerturbationInsertion", methods=["y"], target_class=[1]),
            dict(type="PerturbationDeletion", methods=["x"], target_class=[1]),
        ]})
        self.assertEqual(
            [metric.result_id for metric in metrics],
            ["insertion_0", "insertion_1", "deletion_0"],
        )

    def test_answer_aggregator_prefers_high_insertion_and_low_deletion(self):
        results = {
            "good": {
                "method": "good", "target_class": 1,
                "perturbations": {
                    "insertion_0": {"standard": {"aucs": {"target_probability": 0.9}}},
                    "deletion_0": {"standard": {"aucs": {"target_probability": 0.1}}},
                },
            },
            "bad": {
                "method": "bad", "target_class": 1,
                "perturbations": {
                    "insertion_0": {"standard": {"aucs": {"target_probability": 0.4}}},
                    "deletion_0": {"standard": {"aucs": {"target_probability": 0.8}}},
                },
            },
        }
        answer = FaithfulnessAnswerAggregator().aggregate(results)
        self.assertEqual(answer["winner"]["result_id"], "good")
        self.assertEqual(answer["ranking"][0]["rank"], 1)

    def test_answer_aggregator_selects_metric_ids_and_groups_target_classes(self):
        results = {}
        for target_class in (1, 2):
            results[f"method_class_{target_class}"] = {
                "method": "method", "target_class": target_class,
                "perturbations": {
                    "insertion_0": {"standard": {"aucs": {"target_probability": 0.1}}},
                    "deletion_0": {"standard": {"aucs": {"target_probability": 0.9}}},
                    "insertion_1": {"standard": {"aucs": {"target_probability": 0.8}}},
                    "deletion_1": {"standard": {"aucs": {"target_probability": 0.2}}},
                },
            }
        answer = FaithfulnessAnswerAggregator(
            insertion_metric="insertion_1", deletion_metric="deletion_1"
        ).aggregate(results)
        self.assertIsNone(answer["winner"])
        self.assertEqual(set(answer["winners_by_target_class"]), {"1", "2"})
        self.assertTrue(all(item["final_score"] == 0.8 for item in answer["ranking"]))

    def test_answer_aggregator_rejects_unbounded_curve(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            FaithfulnessAnswerAggregator(curve="target_logit_sum")

    def test_answer_aggregator_rejects_negative_weights(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            FaithfulnessAnswerAggregator(
                insertion_weight=-1.0, deletion_weight=2.0
            )

    def test_registry_wrapper_forwards_dataset_execution_scope(self):
        from experiments.xai_design import RegistryXaiMethod

        method = RegistryXaiMethod(id="legacy_organ", method="organ_occlusion")
        self.assertEqual(method.execution_scope, "dataset")

    def test_legacy_xai_enabled_gate_is_preserved(self):
        self.assertFalse(
            _xai_execution_enabled({"xai": {"enabled": False}}, using_legacy=True)
        )
        self.assertTrue(
            _xai_execution_enabled({"xai": {"enabled": False}}, using_legacy=False)
        )

    def test_organ_occluder_rejects_negative_distances(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            OrganOccluder(feather_mm=-1.0)

    def test_normalization_handles_constant_map(self):
        self.assertTrue(torch.equal(normalize_attribution(torch.ones(2, 2, 2)), torch.zeros(2, 2, 2)))

    def test_signed_normalization_preserves_direction(self):
        result = normalize_signed_attribution(torch.tensor([-2.0, 0.0, 1.0]))
        torch.testing.assert_close(result, torch.tensor([-1.0, 0.0, 0.5]))

    def test_organ_occlusion_builds_signed_map_and_ranking(self):
        source = np.zeros((2, 2, 2), dtype=np.float32)
        source[0, 0, 0] = 2.0
        source[1, 1, 1] = -1.0
        first = np.zeros_like(source, dtype=bool)
        first[0, 0, 0] = True
        second = np.zeros_like(source, dtype=bool)
        second[1, 1, 1] = True
        records = [
            OrganMaskRecord("positive", "Positive", ("positive",), first, np.eye(4)),
            OrganMaskRecord("negative", "Negative", ("negative",), second, np.eye(4)),
        ]

        def preprocess(value, _affine, _label):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32).unsqueeze(0)

        calls = []

        def infer(batch):
            calls.append(batch.clone())
            foreground = batch[:, 0]
            return torch.stack((torch.zeros_like(foreground), foreground), dim=1)

        original = preprocess(source, np.eye(4), False).unsqueeze(0)
        baseline_logits = infer(original)
        calls.clear()
        attribution, details = compute_organ_occlusion_attribution(
            source=source,
            affine=np.eye(4),
            organ_masks=records,
            preprocess_image=preprocess,
            infer=infer,
            baseline_logits=baseline_logits,
            target_class=1,
            objective_id="target_logit_sum",
            method_params={"mode": "fixed_hu", "fill_hu": 0.0, "feather_mm": 0.0},
            device=torch.device("cpu"),
        )

        self.assertEqual(len(calls), 2)
        self.assertGreater(float(attribution[0, 0, 0]), 0.0)
        self.assertLess(float(attribution[1, 1, 1]), 0.0)
        self.assertEqual(details["organs"][0]["organ_id"], "positive")
        self.assertEqual(details["organs"][0]["rank"], 1)
        self.assertEqual(details["organs"][1]["organ_id"], "negative")

    def test_organ_occlusion_rejects_unknown_objective(self):
        source = np.ones((1, 1, 1), dtype=np.float32)
        record = OrganMaskRecord("x", "X", ("x",), source > 0, np.eye(4))

        def preprocess(value, _affine, _label):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32).unsqueeze(0)

        logits = torch.zeros((1, 2, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "Unsupported xai.objective"):
            compute_organ_occlusion_attribution(
                source=source,
                affine=np.eye(4),
                organ_masks=[record],
                preprocess_image=preprocess,
                infer=lambda _batch: logits,
                baseline_logits=logits,
                target_class=1,
                objective_id="unknown",
                method_params={},
                device=torch.device("cpu"),
            )

    def test_organ_occlusion_can_preserve_answer_voxels(self):
        source = np.ones((2, 2, 2), dtype=np.float32)
        record = OrganMaskRecord(
            "organ", "Organ", ("organ",), np.ones_like(source, dtype=bool), np.eye(4)
        )

        def preprocess(value, _affine, _label):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32).unsqueeze(0)

        baseline_input = preprocess(source, np.eye(4), False).unsqueeze(0)
        answer_mask = torch.zeros_like(baseline_input[0, 0], dtype=torch.bool)
        answer_mask[0, 0, 0] = True
        seen = []

        def infer(batch):
            seen.append(batch.clone())
            foreground = batch[:, 0]
            return torch.stack((torch.zeros_like(foreground), foreground), dim=1)

        baseline_logits = infer(baseline_input)
        seen.clear()
        compute_organ_occlusion_attribution(
            source=source,
            affine=np.eye(4),
            organ_masks=[record],
            preprocess_image=preprocess,
            infer=infer,
            baseline_logits=baseline_logits,
            baseline_input=baseline_input,
            answer_mask=answer_mask,
            target_class=1,
            objective_id="target_logit_sum",
            method_params={
                "mode": "fixed_hu",
                "fill_hu": 0.0,
                "feather_mm": 0.0,
                "preserve_answer": True,
            },
            device=torch.device("cpu"),
        )

        self.assertEqual(float(seen[0][0, 0, 0, 0, 0]), 1.0)
        self.assertEqual(float(seen[0][0, 0, 1, 1, 1]), 0.0)

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

    def test_exports_organ_occlusion_ranking(self):
        metrics = {
            "input": "input.nii.gz", "output": "prediction.nii.gz",
            "config": "config.py", "checkpoint": "model.pth", "device": "cpu",
            "class_count": 2, "class_voxels": {"0": 1, "1": 1},
            "benchmark": {
                "latency_seconds": [1.0], "mean_latency_seconds": 1.0,
                "volumes_per_second": 1.0, "parameter_count": 10,
            },
            "xai": {"organ_occlusion_class_1": {
                "method": "organ_occlusion", "target_class": 1,
                "objective": "predicted_mask_dice", "attribution": "organ.nii.gz",
                "perturbations": {},
                "organs": [{
                    "rank": 1, "organ_id": "heart", "display_name": "Heart",
                    "source_labels": ["heart"], "baseline_score": 1.0,
                    "occluded_score": 0.5, "signed_delta": 0.5, "voxel_count": 8,
                }],
            }},
        }
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = export_csv_files(metrics, output_dir)
            ranking = output_dir / "organ_occlusion_ranking.csv"
            self.assertIn(ranking, paths)
            self.assertIn("heart", ranking.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
