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
    predictor: Callable[[Any], Any],
    method: str,
    target_class: int,
    target_mask,
    cfg: Any,
):
    """Compute an input-gradient 3-D XAI map."""
    import torch

    method = str(method).lower()
    supported = {"saliency_map", "input_x_gradient", "smoothgrad"}
    if method not in supported:
        raise ValueError(f"Unsupported experiments XAI method {method!r}; choose {sorted(supported)}")

    samples = int(cfg.xai.smoothgrad_samples) if method == "smoothgrad" else 1
    noise_std = float(cfg.xai.smoothgrad_noise_std) if method == "smoothgrad" else 0.0
    import torch.nn.functional as F

    volume_shape = tuple(batch.shape[2:])
    roi_size = tuple(int(value) for value in cfg.inference.roi_size)
    attribution_sum = torch.zeros_like(batch[0, 0], dtype=torch.float32)

    for _ in range(samples):
        sample = F.interpolate(batch.detach(), size=roi_size, mode="trilinear", align_corners=False)
        if noise_std:
            sample.add_(torch.randn_like(sample) * noise_std)
        sample.requires_grad_(True)
        logits = predictor(sample)
        local_mask = F.interpolate(
            target_mask[None, None].float(), size=roi_size, mode="nearest"
        )[0, 0].bool()
        score = _objective(logits, target_class, local_mask)
        gradient = torch.autograd.grad(score, sample)[0]
        if method == "input_x_gradient":
            attribution = (gradient * sample).abs().amax(dim=1, keepdim=True)
        else:
            attribution = gradient.abs().amax(dim=1, keepdim=True)
        attribution_sum += F.interpolate(
            attribution, size=volume_shape, mode="trilinear", align_corners=False
        )[0, 0].detach()
    return normalize_attribution(attribution_sum / samples)


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
) -> dict[str, Any]:
    """Evaluate attribution faithfulness by inserting/deleting top-ranked voxels."""
    import torch

    fractions = _fractions(int(cfg.metrics.perturbation_steps))
    flat_order = torch.argsort(attribution.flatten(), descending=True)
    voxel_count = flat_order.numel()
    original = batch.detach()
    baseline = torch.full_like(original, float(cfg.metrics.perturbation_baseline))
    insertion_scores: list[float] = []
    deletion_scores: list[float] = []

    with torch.inference_mode():
        for fraction in fractions:
            count = round(fraction * voxel_count)
            selected = flat_order[:count]
            insertion = baseline.clone()
            deletion = original.clone()
            insertion.flatten()[selected] = original.flatten()[selected]
            deletion.flatten()[selected] = baseline.flatten()[selected]
            insertion_scores.append(float(_objective(infer(insertion), target_class, target_mask)))
            deletion_scores.append(float(_objective(infer(deletion), target_class, target_mask)))

    return {
        "fractions": fractions,
        "insertion_scores": insertion_scores,
        "deletion_scores": deletion_scores,
        "insertion_auc": _auc(fractions, insertion_scores),
        "deletion_auc": _auc(fractions, deletion_scores),
    }
