from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil
from typing import Any


def normalize_attribution(attribution):
    """Normalize one 3-D attribution map to [0, 1]."""
    import torch

    attribution = torch.nan_to_num(attribution.float(), nan=0.0, posinf=0.0, neginf=0.0)
    attribution = attribution - attribution.min()
    maximum = attribution.max()
    return attribution / maximum if float(maximum) > 0.0 else torch.zeros_like(attribution)


def target_class_from_prediction(prediction, configured: int | str) -> int:
    """Resolve ``auto`` to the largest predicted foreground class."""
    import torch

    if configured != "auto":
        return int(configured)
    foreground = prediction[prediction > 0]
    if foreground.numel() == 0:
        return 0
    counts = torch.bincount(foreground.to(torch.int64))
    return int(torch.argmax(counts))


def _objective(logits, target_class: int, mask):
    probabilities = logits.softmax(dim=1)[0, target_class]
    if mask is not None and bool(mask.any()):
        return probabilities[mask].mean()
    return probabilities.mean()


def compute_attribution(
    batch,
    model,
    predictor: Callable[[Any], Any],
    method: str,
    target_class: int,
    target_mask,
    cfg: Any,
):
    """Compute a 3-D map with the same XAI registry/methods used by the GUI."""
    import torch
    import torch.nn.functional as F
    from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
    from src.gcd.infrastructure.xai.methods import CamPatchContext, XaiLayerSelection
    from src.gcd.infrastructure.xai.methods.registry import XaiMethodRegistry
    from src.gcd.infrastructure.xai.runtime.layer_hooks import XaiLayerHookManager

    method_id = str(method).lower()
    objective = GradCamEngine._predicted_target_mask_objective
    registry = XaiMethodRegistry.default(objective)
    available = registry.methods_by_id
    if method_id not in available:
        raise ValueError(
            f"Unknown GUI XAI method {method_id!r}; choose {sorted(available)}"
        )
    xai_method = registry.resolve(method_id)
    volume_shape = tuple(batch.shape[2:])
    roi_size = tuple(int(value) for value in cfg.inference.roi_size)
    sample = F.interpolate(
        batch.detach(), size=roi_size, mode="trilinear", align_corners=False
    )
    if xai_method.family == "gradient":
        sample.requires_grad_(True)
    method_params = dict(cfg.xai.get("method_params", {}).get(method_id, {}))
    hook_manager = XaiLayerHookManager(model) if xai_method.uses_layer_controls else None

    model.zero_grad(set_to_none=True)
    if hook_manager is None:
        logits = predictor(sample)
        layers_by_name = {}
    else:
        with hook_manager:
            logits = predictor(sample)
            layers_by_name = hook_manager.layers_by_name()
            payload = xai_method.collect_patch_data(
                CamPatchContext(
                    input_tensor=sample,
                    logits=logits,
                    layers_by_name=layers_by_name,
                    target_class=target_class,
                    objective=objective,
                    model=model,
                    method_params=method_params,
                    device=batch.device,
                )
            )
    if hook_manager is None:
        payload = xai_method.collect_patch_data(
            CamPatchContext(
                input_tensor=sample,
                logits=logits,
                layers_by_name=layers_by_name,
                target_class=target_class,
                objective=objective,
                model=model,
                method_params=method_params,
                device=batch.device,
            )
        )

    if xai_method.uses_layer_controls:
        requested_layer = str(cfg.xai.get("layer", "") or cfg.get("default_layer", ""))
        layer = requested_layer if requested_layer in layers_by_name else next(iter(layers_by_name))
        channel_count = int(layers_by_name[layer].shape[1])
    else:
        layer = "input"
        channel_count = 1
    attribution = xai_method.build_tile_cam(
        payload,
        XaiLayerSelection(layer, 0, channel_count, roi_size),
        method_params=method_params,
    )
    if xai_method.family == "gradient":
        attribution = torch.relu(attribution)
    attribution = F.interpolate(
        attribution, size=volume_shape, mode="trilinear", align_corners=False
    )[0, 0]
    return normalize_attribution(attribution.detach())


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


def _binary_overlap(prediction, mask) -> tuple[float, float]:
    prediction = prediction.bool()
    mask = mask.bool()
    intersection = int((prediction & mask).sum())
    prediction_count = int(prediction.sum())
    mask_count = int(mask.sum())
    union = prediction_count + mask_count - intersection
    dice = 1.0 if prediction_count + mask_count == 0 else 2.0 * intersection / (prediction_count + mask_count)
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


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
    """One independently configured insertion or deletion experiment."""

    config: Any
    operation: str = ""
    config_index: int = 0

    @property
    def result_id(self) -> str:
        return f"{self.operation}_{self.config_index}"

    def progress_steps(self, answer_mask=None) -> int:
        """Number of inference steps that will actually run."""
        variants = 0
        answer_exists = answer_mask is not None and int(answer_mask.sum()) > 0
        for retention in _retention_values(self.config):
            if retention is None or answer_exists:
                variants += 1
        return variants * (int(self.config.get("steps", 20)) + 1)

    def _initial_and_order(self, original, baseline, flat_order, answer_mask, retention):
        import torch

        if retention is None:
            return self._initial(original, baseline), flat_order
        answer_flat = answer_mask.flatten().bool()
        answer_indices = torch.nonzero(answer_flat, as_tuple=False).flatten()
        keep_count = ceil(retention * answer_indices.numel())
        # Choose the retained subset independently of the attribution method so
        # answer-preserving curves remain comparable between methods.
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
        baseline = torch.full_like(original, float(self.config.get("baseline", 0.0)))
        flat_order = torch.argsort(attribution.flatten(), descending=True)
        variants: dict[str, Any] = {}
        original_prediction = target_mask.bool()

        for retention in _retention_values(self.config):
            key = "standard" if retention is None else f"answer_retention_{retention:g}"
            if retention is not None and answer_mask is None:
                variants[key] = {
                    "status": "skipped",
                    "reason": "answer-preserving perturbation requires ground truth",
                    "answer_retention": retention,
                }
                continue
            if retention is not None and int(answer_mask.sum()) == 0:
                variants[key] = {
                    "status": "skipped",
                    "reason": "target class is absent from ground truth",
                    "answer_retention": retention,
                }
                continue
            initial, order = self._initial_and_order(
                original, baseline, flat_order, answer_mask, retention
            )
            scores: list[float] = []
            prediction_dice: list[float] = []
            prediction_iou: list[float] = []
            ground_truth_dice: list[float] = []
            ground_truth_iou: list[float] = []
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
                    scores.append(float(_objective(logits, target_class, target_mask)))
                    perturbed_prediction = torch.argmax(logits, dim=1)[0] == target_class
                    dice, iou = _binary_overlap(perturbed_prediction, original_prediction)
                    prediction_dice.append(dice)
                    prediction_iou.append(iou)
                    if answer_mask is not None:
                        dice, iou = _binary_overlap(perturbed_prediction, answer_mask)
                        ground_truth_dice.append(dice)
                        ground_truth_iou.append(iou)
                    if progress_callback is not None:
                        progress_callback()
            variants[key] = {
                "status": "completed",
                "answer_retention": retention,
                "fractions": fractions,
                "scores": scores,
                "auc": _auc(fractions, scores),
                "prediction_dice": prediction_dice,
                "prediction_iou": prediction_iou,
                "ground_truth_dice": ground_truth_dice,
                "ground_truth_iou": ground_truth_iou,
            }
        return variants


class PerturbationInsertion(_PerturbationMetric):
    operation = "insertion"

    def __init__(self, config: Any, config_index: int = 0):
        super().__init__(
            config=config, operation="insertion", config_index=config_index
        )

    def _initial(self, original, baseline):
        return baseline.clone()

    def _apply(self, perturbed, original, baseline, selected) -> None:
        perturbed.flatten()[selected] = original.flatten()[selected]


class PerturbationDeletion(_PerturbationMetric):
    operation = "deletion"

    def __init__(self, config: Any, config_index: int = 0):
        super().__init__(
            config=config, operation="deletion", config_index=config_index
        )

    def _initial(self, original, baseline):
        return original.clone()

    def _apply(self, perturbed, original, baseline, selected) -> None:
        perturbed.flatten()[selected] = baseline.flatten()[selected]


def configured_perturbations(cfg: Any) -> Iterable[_PerturbationMetric]:
    """Presence of a config block enables that operation; no booleans needed."""
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
        result.update(insertion_scores=curve["scores"], insertion_auc=curve["auc"])
    if bool(getattr(metrics, "deletion_enabled", True)):
        curve = PerturbationDeletion(config).evaluate(
            batch,
            attribution,
            infer,
            target_class,
            target_mask,
            progress=progress,
        )["standard"]
        result.update(deletion_scores=curve["scores"], deletion_auc=curve["auc"])
    return result
