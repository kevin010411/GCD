from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil
from typing import Any

from ..xai_design import XAI_METRICS, XAI_SCORERS


def _fractions(steps: int) -> list[float]:
    if steps < 1:
        raise ValueError("metrics.perturbation_steps must be at least 1")
    return [index / steps for index in range(steps + 1)]


def _auc(fractions: list[float], scores: list[float]) -> float:
    return sum(
        (scores[index] + scores[index + 1])
        * (fractions[index + 1] - fractions[index])
        / 2.0
        for index in range(len(scores) - 1)
    )


def _retention_values(config: Any) -> list[float | None]:
    values = config.get("answer_retention", (None,))
    if values is None or isinstance(values, (int, float)):
        values = (values,)
    result: list[float | None] = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        retention = float(value)
        if not 0.0 <= retention <= 1.0:
            raise ValueError("answer_retention values must be between 0 and 1")
        result.append(retention)
    return result


@dataclass
class _PerturbationMetric:
    """Template method for one independently configured perturbation experiment."""

    config: Any
    operation: str = ""
    config_index: int = 0

    def __post_init__(self) -> None:
        scorer_configs = self.config.get("scorers") or (
            dict(type="TargetProbabilityScore"),
            dict(type="PredictionDiceScore"),
            dict(type="PredictionIoUScore"),
            dict(type="GroundTruthDiceScore"),
            dict(type="GroundTruthIoUScore"),
        )
        self.scorers = [
            XAI_SCORERS.build(item) if isinstance(item, dict) else item
            for item in scorer_configs
        ]
        scorer_ids = [scorer.id for scorer in self.scorers]
        if len(set(scorer_ids)) != len(scorer_ids):
            raise ValueError(
                "A perturbation metric cannot contain duplicate scorer ids"
            )

    @property
    def result_id(self) -> str:
        return f"{self.operation}_{self.config_index}"

    def progress_steps(self, answer_mask=None) -> int:
        variants = 0
        answer_exists = answer_mask is not None and int(answer_mask.sum()) > 0
        for retention in _retention_values(self.config):
            if retention is None or answer_exists:
                variants += 1
        return variants * (int(self.config.get("steps", 20)) + 1)

    def _initial_and_order(
        self, original, baseline, flat_order, answer_mask, retention
    ):
        import torch

        if retention is None:
            return self._initial(original, baseline), flat_order
        answer_flat = answer_mask.flatten().bool()
        answer_indices = torch.nonzero(answer_flat, as_tuple=False).flatten()
        keep_count = ceil(retention * answer_indices.numel())
        retained = answer_indices[:keep_count]
        protected = torch.zeros_like(answer_flat)
        protected[retained] = True
        eligible_order = flat_order[~protected[flat_order]]
        initial = self._initial(original, baseline)
        initial.flatten()[retained] = original.flatten()[retained]
        return initial, eligible_order

    def _initial(self, original, baseline):
        raise NotImplementedError

    def _apply(self, perturbed, original, baseline, selected) -> None:
        raise NotImplementedError

    def evaluate(
        self,
        batch,
        attribution,
        infer: Callable[[Any], Any],
        target_class: int,
        target_mask,
        answer_mask=None,
        *,
        progress: bool = True,
        progress_callback: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        import torch
        from tqdm.auto import tqdm

        fractions = _fractions(int(self.config.get("steps", 20)))
        original = batch.detach()
        baseline = torch.full_like(
            original, float(self.config.get("baseline", 0.0))
        )
        flat_order = torch.argsort(attribution.flatten(), descending=True)
        variants: dict[str, Any] = {}
        original_prediction = target_mask.bool()

        for retention in _retention_values(self.config):
            key = (
                "standard"
                if retention is None
                else f"answer_retention_{retention:g}"
            )
            skipped = self._skip_reason(retention, answer_mask)
            if skipped is not None:
                variants[key] = {
                    "status": "skipped",
                    "reason": skipped,
                    "answer_retention": retention,
                }
                continue
            initial, order = self._initial_and_order(
                original, baseline, flat_order, answer_mask, retention
            )
            curve_values = {scorer.id: [] for scorer in self.scorers}
            with torch.inference_mode():
                for fraction in tqdm(
                    fractions,
                    desc=f"{self.operation} {key}",
                    unit="step",
                    leave=False,
                    disable=not progress,
                ):
                    selected = order[: round(fraction * order.numel())]
                    perturbed = initial.clone()
                    self._apply(perturbed, original, baseline, selected)
                    logits = infer(perturbed)
                    for scorer in self.scorers:
                        value = scorer.score(
                            logits,
                            target_class,
                            original_prediction,
                            answer_mask,
                        )
                        if value is not None:
                            curve_values[scorer.id].append(float(value))
                    if progress_callback is not None:
                        progress_callback()
            variants[key] = self._completed_variant(
                retention, fractions, curve_values
            )
        return variants

    @staticmethod
    def _skip_reason(retention, answer_mask) -> str | None:
        if retention is not None and answer_mask is None:
            return "answer-preserving perturbation requires ground truth"
        if retention is not None and int(answer_mask.sum()) == 0:
            return "target class is absent from ground truth"
        return None

    @staticmethod
    def _completed_variant(retention, fractions, curve_values):
        aucs = {
            scorer_id: _auc(fractions, values)
            for scorer_id, values in curve_values.items()
            if len(values) == len(fractions)
        }
        scores = curve_values.get("target_probability", [])
        return {
            "status": "completed",
            "answer_retention": retention,
            "fractions": fractions,
            "scores": scores,
            "auc": aucs.get("target_probability"),
            "aucs": aucs,
            "curves": curve_values,
            "prediction_dice": curve_values.get("prediction_dice", []),
            "prediction_iou": curve_values.get("prediction_iou", []),
            "ground_truth_dice": curve_values.get("ground_truth_dice", []),
            "ground_truth_iou": curve_values.get("ground_truth_iou", []),
        }


@XAI_METRICS.register_module()
class PerturbationInsertion(_PerturbationMetric):
    operation = "insertion"

    def __init__(self, config: Any = None, config_index: int = 0, **kwargs):
        config = dict(config or {})
        config.update(kwargs)
        super().__init__(
            config=config, operation="insertion", config_index=config_index
        )

    def _initial(self, original, baseline):
        return baseline.clone()

    def _apply(self, perturbed, original, baseline, selected) -> None:
        perturbed.flatten()[selected] = original.flatten()[selected]


@XAI_METRICS.register_module()
class PerturbationDeletion(_PerturbationMetric):
    operation = "deletion"

    def __init__(self, config: Any = None, config_index: int = 0, **kwargs):
        config = dict(config or {})
        config.update(kwargs)
        super().__init__(
            config=config, operation="deletion", config_index=config_index
        )

    def _initial(self, original, baseline):
        return original.clone()

    def _apply(self, perturbed, original, baseline, selected) -> None:
        perturbed.flatten()[selected] = baseline.flatten()[selected]


def configured_perturbations(cfg: Any) -> Iterable[_PerturbationMetric]:
    """Build perturbations from the legacy top-level config blocks."""
    insertion_configs = cfg.get("PerturbationInsertion", ())
    deletion_configs = cfg.get("PerturbationDeletion", ())
    if hasattr(insertion_configs, "get"):
        insertion_configs = (insertion_configs,)
    if hasattr(deletion_configs, "get"):
        deletion_configs = (deletion_configs,)
    for index, config in enumerate(insertion_configs or ()):
        yield PerturbationInsertion(config, index)
    for index, config in enumerate(deletion_configs or ()):
        yield PerturbationDeletion(config, index)


def insertion_deletion_metrics(
    batch,
    attribution,
    infer: Callable[[Any], Any],
    target_class: int,
    target_mask,
    cfg: Any,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Compatibility adapter for callers using the former combined API."""
    metrics = cfg.metrics
    config = {
        "steps": int(metrics.perturbation_steps),
        "baseline": float(metrics.perturbation_baseline),
        "answer_retention": (None,),
    }
    result = {"fractions": _fractions(config["steps"])}
    if bool(getattr(metrics, "insertion_enabled", True)):
        curve = PerturbationInsertion(config).evaluate(
            batch,
            attribution,
            infer,
            target_class,
            target_mask,
            progress=progress,
        )["standard"]
        result.update(
            insertion_scores=curve["scores"], insertion_auc=curve["auc"]
        )
    if bool(getattr(metrics, "deletion_enabled", True)):
        curve = PerturbationDeletion(config).evaluate(
            batch,
            attribution,
            infer,
            target_class,
            target_mask,
            progress=progress,
        )["standard"]
        result.update(
            deletion_scores=curve["scores"], deletion_auc=curve["auc"]
        )
    return result
