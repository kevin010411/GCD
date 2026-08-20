from __future__ import annotations

from collections.abc import Callable
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
    """Evaluate attribution faithfulness by inserting/deleting top-ranked voxels."""
    import torch

    fractions = _fractions(int(cfg.metrics.perturbation_steps))
    flat_order = torch.argsort(attribution.flatten(), descending=True)
    voxel_count = flat_order.numel()
    original = batch.detach()
    baseline = torch.full_like(original, float(cfg.metrics.perturbation_baseline))
    insertion_enabled = bool(getattr(cfg.metrics, "insertion_enabled", True))
    deletion_enabled = bool(getattr(cfg.metrics, "deletion_enabled", True))
    insertion_scores: list[float] = []
    deletion_scores: list[float] = []

    with torch.inference_mode():
        from tqdm.auto import tqdm

        for fraction in tqdm(
            fractions,
            desc="Insertion/deletion",
            unit="step",
            leave=False,
            disable=not progress,
        ):
            count = round(fraction * voxel_count)
            selected = flat_order[:count]
            if insertion_enabled:
                insertion = baseline.clone()
                insertion.flatten()[selected] = original.flatten()[selected]
                insertion_scores.append(float(_objective(infer(insertion), target_class, target_mask)))
            if deletion_enabled:
                deletion = original.clone()
                deletion.flatten()[selected] = baseline.flatten()[selected]
                deletion_scores.append(float(_objective(infer(deletion), target_class, target_mask)))

    result = {
        "fractions": fractions,
    }
    if insertion_enabled:
        result.update(insertion_scores=insertion_scores, insertion_auc=_auc(fractions, insertion_scores))
    if deletion_enabled:
        result.update(deletion_scores=deletion_scores, deletion_auc=_auc(fractions, deletion_scores))
    return result
