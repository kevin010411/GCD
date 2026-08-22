from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from mmengine import Registry


XAI_METHODS = Registry("experiment_xai_method")
XAI_SCORERS = Registry("experiment_xai_scorer")
XAI_METRICS = Registry("experiment_xai_metric")
XAI_ANSWERS = Registry("experiment_xai_answer")
XAI_COMPONENTS = Registry("experiment_xai_component")


@dataclass
class XaiExecutionContext:
    batch: Any
    model: Any
    predictor: Any
    target_class: int
    target_mask: Any
    cfg: Any
    dataset_context: dict[str, Any]


class ExperimentXaiMethod(ABC):
    execution_scope = "sample"

    def __init__(self, id: str, target_class: Any = "auto") -> None:
        self.id = str(id)
        values = target_class if isinstance(target_class, (list, tuple)) else [target_class]
        self.target_classes = tuple(values or ("auto",))

    @abstractmethod
    def explain(self, context: XaiExecutionContext) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError


class ScoreMetric(ABC):
    id = ""

    @abstractmethod
    def score(
        self,
        logits,
        target_class: int,
        reference_prediction,
        ground_truth=None,
    ) -> float | None:
        raise NotImplementedError


@XAI_COMPONENTS.register_module()
class TotalSegmentatorService:
    def __init__(
        self,
        cache_root: str = "output/organ_masks",
        task: str = "total",
        merge_organs: bool = True,
    ):
        from src.gcd.infrastructure.organ_occlusion import TotalSegmentatorOrganService

        self.service = TotalSegmentatorOrganService(cache_root)
        self.task = str(task)
        self.merge_organs = bool(merge_organs)

    def run(self, input_path, device: str):
        return self.service.run(
            input_path,
            device=device,
            task=self.task,
            merge_organs=self.merge_organs,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self.service.last_run_metadata


@XAI_COMPONENTS.register_module()
class OrganOccluder:
    def __init__(
        self,
        mode: str = "local_mean",
        fill_hu: float = 0.0,
        feather_mm: float = 2.0,
        blur_sigma_mm: float = 3.0,
        local_mean_shell_mm: float = 5.0,
        preserve_answer: bool = False,
    ) -> None:
        if mode not in {"local_mean", "gaussian_blur", "fixed_hu"}:
            raise ValueError(f"Unsupported organ occlusion mode: {mode}")
        if mode == "fixed_hu" and not -1000.0 <= float(fill_hu) <= 1000.0:
            raise ValueError("fixed_hu fill_hu must be between -1000 and 1000")
        radii = (float(feather_mm), float(blur_sigma_mm), float(local_mean_shell_mm))
        if any(value < 0.0 for value in radii):
            raise ValueError("Organ occlusion distance parameters must be non-negative")
        self.mode = mode
        self.fill_hu = float(fill_hu)
        self.feather_mm, self.blur_sigma_mm, self.local_mean_shell_mm = radii
        self.preserve_answer = bool(preserve_answer)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "fill_hu": self.fill_hu,
            "feather_mm": self.feather_mm,
            "blur_sigma_mm": self.blur_sigma_mm,
            "local_mean_shell_mm": self.local_mean_shell_mm,
            "preserve_answer": self.preserve_answer,
        }

    def apply(self, source, records_by_id, organ_id: str, spacing):
        from src.gcd.domain import OrganIntervention, OrganOcclusionSpec
        from src.gcd.infrastructure.organ_occlusion import apply_organ_occlusion

        spec = OrganOcclusionSpec(
            interventions=(OrganIntervention(
                organ_id, enabled=True, mode=self.mode, fill_hu=self.fill_hu
            ),),
            feather_mm=self.feather_mm,
            blur_sigma_mm=self.blur_sigma_mm,
            local_mean_shell_mm=self.local_mean_shell_mm,
        )
        return apply_organ_occlusion(source, records_by_id, spec, spacing=spacing)


def _binary_overlap(prediction, reference) -> tuple[float, float]:
    import torch

    prediction = prediction.bool()
    reference = reference.to(device=prediction.device, dtype=torch.bool)
    intersection = torch.logical_and(prediction, reference).sum(dtype=torch.float32)
    total = prediction.sum(dtype=torch.float32) + reference.sum(dtype=torch.float32)
    union = torch.logical_or(prediction, reference).sum(dtype=torch.float32)
    dice = 1.0 if float(total) == 0.0 else float(2.0 * intersection / total)
    iou = 1.0 if float(union) == 0.0 else float(intersection / union)
    return dice, iou


@XAI_SCORERS.register_module()
class TargetProbabilityScore(ScoreMetric):
    id = "target_probability"

    def score(self, logits, target_class, reference_prediction, ground_truth=None):
        probabilities = logits.softmax(dim=1)[0, target_class]
        mask = reference_prediction.to(device=probabilities.device, dtype=bool)
        return float(probabilities[mask].mean()) if bool(mask.any()) else float(probabilities.mean())


@XAI_SCORERS.register_module()
class TargetProbabilitySumScore(ScoreMetric):
    id = "target_probability_sum"

    def score(self, logits, target_class, reference_prediction, ground_truth=None):
        return float(logits.softmax(dim=1)[0, target_class].sum())


@XAI_SCORERS.register_module()
class TargetLogitSumScore(ScoreMetric):
    id = "target_logit_sum"

    def score(self, logits, target_class, reference_prediction, ground_truth=None):
        return float(logits[0, target_class].sum())


@XAI_SCORERS.register_module()
class PredictionDiceScore(ScoreMetric):
    id = "prediction_dice"

    def score(self, logits, target_class, reference_prediction, ground_truth=None):
        prediction = logits.argmax(dim=1)[0] == target_class
        return _binary_overlap(prediction, reference_prediction)[0]


@XAI_SCORERS.register_module()
class PredictionIoUScore(ScoreMetric):
    id = "prediction_iou"

    def score(self, logits, target_class, reference_prediction, ground_truth=None):
        prediction = logits.argmax(dim=1)[0] == target_class
        return _binary_overlap(prediction, reference_prediction)[1]


@XAI_SCORERS.register_module()
class GroundTruthDiceScore(ScoreMetric):
    id = "ground_truth_dice"

    def score(self, logits, target_class, reference_prediction, ground_truth=None):
        if ground_truth is None:
            return None
        prediction = logits.argmax(dim=1)[0] == target_class
        return _binary_overlap(prediction, ground_truth)[0]


@XAI_SCORERS.register_module()
class GroundTruthIoUScore(ScoreMetric):
    id = "ground_truth_iou"

    def score(self, logits, target_class, reference_prediction, ground_truth=None):
        if ground_truth is None:
            return None
        prediction = logits.argmax(dim=1)[0] == target_class
        return _binary_overlap(prediction, ground_truth)[1]


@XAI_METHODS.register_module()
class RegistryXaiMethod(ExperimentXaiMethod):
    def __init__(
        self,
        id: str,
        method: str | None = None,
        layer: str = "",
        params: dict[str, Any] | None = None,
        target_class: Any = "auto",
    ) -> None:
        super().__init__(id, target_class)
        self.method = str(method or id)
        self.layer = str(layer)
        self.params = dict(params or {})
        from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
        from src.gcd.infrastructure.xai.methods.registry import XaiMethodRegistry

        registry = XaiMethodRegistry.default(
            GradCamEngine._predicted_target_mask_objective
        )
        if self.method not in registry.methods_by_id:
            raise ValueError(f"Unknown registered XAI method: {self.method}")
        self.delegate = registry.resolve(self.method)

    @property
    def execution_scope(self) -> str:
        return str(getattr(self.delegate, "execution_scope", "sample"))

    def explain(self, context: XaiExecutionContext):
        from .xai import compute_attribution

        return compute_attribution(
            context.batch,
            context.model,
            context.predictor,
            self.method,
            context.target_class,
            context.target_mask,
            context.cfg,
            dataset_context=context.dataset_context,
            return_metadata=True,
            method_params=self.params,
            layer=self.layer,
            xai_method_override=self.delegate,
        )


@XAI_METHODS.register_module()
class OrganOcclusionXaiMethod(ExperimentXaiMethod):
    execution_scope = "dataset"

    def __init__(
        self,
        id: str = "organ_occlusion",
        objective: dict[str, Any] | ScoreMetric | None = None,
        segmenter: dict[str, Any] | Any | None = None,
        occluder: dict[str, Any] | Any | None = None,
        target_class: Any = "auto",
    ) -> None:
        super().__init__(id, target_class)
        self.objective = (
            XAI_SCORERS.build(objective or dict(type="PredictionDiceScore"))
            if isinstance(objective, dict) or objective is None
            else objective
        )
        self.segmenter = (
            XAI_COMPONENTS.build(segmenter or dict(type="TotalSegmentatorService"))
            if isinstance(segmenter, dict) or segmenter is None
            else segmenter
        )
        self.occluder = (
            XAI_COMPONENTS.build(occluder or dict(type="OrganOccluder"))
            if isinstance(occluder, dict) or occluder is None
            else occluder
        )

    def explain(self, context: XaiExecutionContext):
        from .xai import compute_organ_occlusion_attribution

        runtime = context.dataset_context
        device_name = "gpu" if str(context.batch.device).startswith("cuda") else "cpu"
        organ_masks = self.segmenter.run(runtime["input_path"], device_name)
        progress_start = runtime.get("progress_start_callback")
        if progress_start is not None:
            progress_start(len(organ_masks) + 1)
        progress_callback = runtime.get("progress_callback")
        if progress_callback is not None:
            progress_callback(1)
        attribution, metadata = compute_organ_occlusion_attribution(
            source=runtime["source"],
            affine=runtime["affine"],
            organ_masks=organ_masks,
            preprocess_image=runtime["preprocess_image"],
            infer=runtime["infer"],
            baseline_logits=runtime["baseline_logits"],
            baseline_input=context.batch,
            answer_mask=runtime.get("answer_mask", context.target_mask),
            target_class=context.target_class,
            objective_id=self.objective.id,
            method_params=self.occluder.parameters,
            device=context.batch.device,
            progress_callback=progress_callback,
            objective=self.objective,
            occluder=self.occluder,
        )
        metadata["totalsegmentator"] = self.segmenter.metadata
        return attribution, metadata


@XAI_ANSWERS.register_module()
class FaithfulnessAnswerAggregator:
    """Rank explanations using insertion-high/deletion-low faithfulness AUC."""

    def __init__(
        self,
        insertion_weight: float = 0.5,
        deletion_weight: float = 0.5,
        curve: str = "target_probability",
        variant: str = "standard",
        insertion_metric: str = "insertion_0",
        deletion_metric: str = "deletion_0",
    ) -> None:
        if float(insertion_weight) < 0.0 or float(deletion_weight) < 0.0:
            raise ValueError("Answer weights must be non-negative")
        total = float(insertion_weight) + float(deletion_weight)
        if total <= 0:
            raise ValueError("Answer weights must have a positive sum")
        self.insertion_weight = float(insertion_weight) / total
        self.deletion_weight = float(deletion_weight) / total
        self.curve = str(curve)
        self.variant = str(variant)
        bounded = {
            "target_probability", "prediction_dice", "prediction_iou",
            "ground_truth_dice", "ground_truth_iou",
        }
        if self.curve not in bounded:
            raise ValueError(
                f"FaithfulnessAnswerAggregator curve must be bounded in [0, 1]: {self.curve}"
            )
        self.insertion_metric = str(insertion_metric)
        self.deletion_metric = str(deletion_metric)

    def aggregate(self, xai_results: dict[str, Any]) -> dict[str, Any]:
        rankings = []
        for result_id, result in xai_results.items():
            insertion_auc = self._auc(result, "insertion")
            deletion_auc = self._auc(result, "deletion")
            if insertion_auc is None or deletion_auc is None:
                continue
            final_score = (
                self.insertion_weight * insertion_auc
                + self.deletion_weight * (1.0 - deletion_auc)
            )
            rankings.append(
                {
                    "result_id": result_id,
                    "method": result["method"],
                    "target_class": result["target_class"],
                    "insertion_auc": insertion_auc,
                    "deletion_auc": deletion_auc,
                    "final_score": final_score,
                }
            )
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in rankings:
            grouped.setdefault(int(item["target_class"]), []).append(item)
        winners = {}
        rankings = []
        for target_class, items in sorted(grouped.items()):
            items.sort(key=lambda item: item["final_score"], reverse=True)
            for rank, item in enumerate(items, start=1):
                item["rank"] = rank
            winners[str(target_class)] = items[0]
            rankings.extend(items)
        return {
            "metric": "faithfulness",
            "curve": self.curve,
            "variant": self.variant,
            "weights": {
                "insertion": self.insertion_weight,
                "deletion": self.deletion_weight,
            },
            "winner": rankings[0] if len(grouped) == 1 and rankings else None,
            "winners_by_target_class": winners,
            "ranking": rankings,
        }

    def _auc(self, result: dict[str, Any], operation: str) -> float | None:
        metric_id = (
            self.insertion_metric if operation == "insertion" else self.deletion_metric
        )
        variants = result.get("perturbations", {}).get(metric_id)
        if variants is None:
            return None
        curve = variants.get(self.variant, {})
        aucs = curve.get("aucs", {})
        value = aucs.get(self.curve)
        if value is None and self.curve == "target_probability":
            value = curve.get("auc")
        return float(value) if value is not None else None


def build_xai_methods(cfg: Any) -> dict[str, ExperimentXaiMethod]:
    methods = [XAI_METHODS.build(item) for item in cfg.get("XaiMethods", ())]
    methods_by_id = {method.id: method for method in methods}
    if len(methods_by_id) != len(methods):
        raise ValueError("XaiMethods contains duplicate method ids")
    return methods_by_id


def build_xai_metrics(cfg: Any) -> list[Any]:
    # Importing the module registers the concrete template-method strategies.
    from . import xai as _registered_metrics  # noqa: F401

    counters: dict[str, int] = {}
    metrics = []
    for item in cfg.get("XaiMetrics", ()):
        metric_cfg = dict(item)
        type_name = str(metric_cfg.get("type", ""))
        operation = {
            "PerturbationInsertion": "insertion",
            "PerturbationDeletion": "deletion",
        }.get(type_name, type_name)
        metric_cfg.setdefault("config_index", counters.get(operation, 0))
        counters[operation] = counters.get(operation, 0) + 1
        metrics.append(XAI_METRICS.build(metric_cfg))
    return metrics


def build_xai_answer(cfg: Any):
    answer_cfg = cfg.get("XaiAnswer")
    return XAI_ANSWERS.build(answer_cfg) if answer_cfg else None
