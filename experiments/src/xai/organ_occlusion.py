from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .normalization import normalize_signed_attribution


SUPPORTED_DATASET_OBJECTIVES = {
    "predicted_mask_dice",
    "predicted_mask_iou",
    "target_probability_sum",
    "target_logit_sum",
}


def _dataset_objective(
    logits, target_class: int, objective_id: str, reference_mask, objective=None
):
    import torch

    if objective is not None:
        return float(objective.score(logits, target_class, reference_mask))
    if objective_id not in SUPPORTED_DATASET_OBJECTIVES:
        raise ValueError(
            f"Unsupported xai.objective {objective_id!r}; choose "
            f"{sorted(SUPPORTED_DATASET_OBJECTIVES)}"
        )
    if objective_id == "target_probability_sum":
        return float(logits.softmax(dim=1)[0, target_class].sum())
    if objective_id == "target_logit_sum":
        return float(logits[0, target_class].sum())
    prediction = torch.argmax(logits, dim=1)[0] == target_class
    reference = reference_mask.to(device=prediction.device, dtype=torch.bool)
    intersection = torch.logical_and(
        prediction, reference
    ).sum(dtype=torch.float32)
    if objective_id == "predicted_mask_dice":
        denominator = prediction.sum(dtype=torch.float32) + reference.sum(
            dtype=torch.float32
        )
        return (
            1.0
            if float(denominator) == 0.0
            else float(2.0 * intersection / denominator)
        )
    union = torch.logical_or(prediction, reference).sum(dtype=torch.float32)
    return 1.0 if float(union) == 0.0 else float(intersection / union)


def compute_organ_occlusion_attribution(
    *,
    source,
    affine,
    organ_masks,
    preprocess_image: Callable[[Any, Any, bool], Any],
    infer: Callable[[Any], Any],
    baseline_logits,
    target_class: int,
    objective_id: str,
    method_params: dict[str, Any],
    device,
    baseline_input=None,
    answer_mask=None,
    progress_callback: Callable[[int], None] | None = None,
    objective=None,
    occluder=None,
):
    """Evaluate every TotalSegmentator organ and build a signed 3-D map."""
    import numpy as np
    import torch
    from src.gcd.domain import OrganIntervention, OrganOcclusionSpec
    from src.gcd.infrastructure.organ_occlusion import (
        apply_organ_occlusion,
        validate_mask_geometry,
    )

    mode = str(method_params.get("mode", "local_mean"))
    if mode not in {"local_mean", "gaussian_blur", "fixed_hu"}:
        raise ValueError(
            f"Unsupported organ occlusion mode {mode!r}; choose local_mean, "
            "gaussian_blur, or fixed_hu"
        )
    fill_hu = float(method_params.get("fill_hu", 0.0))
    if mode == "fixed_hu" and not -1000.0 <= fill_hu <= 1000.0:
        raise ValueError("fixed_hu fill_hu must be between -1000 and 1000")
    if not organ_masks:
        raise ValueError("TotalSegmentator returned no non-empty organ masks")

    reference_mask = torch.argmax(baseline_logits, dim=1)[0] == int(target_class)
    baseline_score = _dataset_objective(
        baseline_logits, target_class, objective_id, reference_mask, objective
    )
    output_shape = tuple(int(value) for value in baseline_logits.shape[2:])
    weighted = torch.zeros(output_shape, dtype=torch.float32)
    counts = torch.zeros(output_shape, dtype=torch.float32)
    spacing = tuple(
        float(value if value > 0 else 1.0)
        for value in np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    )
    records_by_id = {record.id: record for record in organ_masks}
    rankings: list[dict[str, Any]] = []
    spec_options = {
        "feather_mm": float(method_params.get("feather_mm", 2.0)),
        "blur_sigma_mm": float(method_params.get("blur_sigma_mm", 3.0)),
        "local_mean_shell_mm": float(
            method_params.get("local_mean_shell_mm", 5.0)
        ),
    }
    preserve_answer = bool(method_params.get("preserve_answer", False))
    if preserve_answer and (baseline_input is None or answer_mask is None):
        raise ValueError(
            "preserve_answer requires both baseline_input and answer_mask"
        )
    if any(value < 0.0 for value in spec_options.values()):
        raise ValueError("Organ occlusion distance parameters must be non-negative")

    for record in organ_masks:
        validate_mask_geometry(source, affine, record.mask, record.affine)
        original_mask = np.asarray(record.mask, dtype=bool)
        if not np.any(original_mask):
            continue
        if occluder is None:
            spec = OrganOcclusionSpec(
                interventions=(
                    OrganIntervention(
                        record.id, enabled=True, mode=mode, fill_hu=fill_hu
                    ),
                ),
                **spec_options,
            )
            occluded, _modified = apply_organ_occlusion(
                source, records_by_id, spec, spacing=spacing
            )
        else:
            occluded, _modified = occluder.apply(
                source, records_by_id, record.id, spacing
            )
        perturbed = preprocess_image(occluded, affine, False).to(
            device=device, dtype=torch.float32
        ).unsqueeze(0)
        if preserve_answer:
            protected = answer_mask.to(device=device, dtype=torch.bool)
            if tuple(protected.shape) != tuple(perturbed.shape[2:]):
                raise ValueError(
                    "Answer mask shape does not match the preprocessed input shape"
                )
            original = baseline_input.to(device=device, dtype=perturbed.dtype)
            perturbed[:, :, protected] = original[:, :, protected]
        with torch.inference_mode():
            perturbed_logits = infer(perturbed)
        occluded_score = _dataset_objective(
            perturbed_logits,
            target_class,
            objective_id,
            reference_mask,
            objective,
        )
        delta = baseline_score - occluded_score
        transformed_mask = preprocess_image(
            original_mask.astype(np.float32), affine, True
        )[0].to(dtype=torch.bool, device="cpu")
        if tuple(transformed_mask.shape) != output_shape:
            raise ValueError(
                f"Preprocessed organ mask {record.id!r} shape "
                f"{tuple(transformed_mask.shape)} does not match logits {output_shape}"
            )
        if preserve_answer:
            transformed_mask &= ~answer_mask.to(dtype=torch.bool, device="cpu")
        weighted[transformed_mask] += float(delta)
        counts[transformed_mask] += 1.0
        rankings.append(
            {
                "organ_id": record.id,
                "display_name": record.display_name,
                "source_labels": list(record.source_labels),
                "baseline_score": baseline_score,
                "occluded_score": occluded_score,
                "signed_delta": float(delta),
                "voxel_count": int(original_mask.sum()),
            }
        )
        if progress_callback is not None:
            progress_callback(1)

    if not rankings:
        raise ValueError("TotalSegmentator returned no non-empty organ masks")
    attribution = torch.where(
        counts > 0, weighted / counts.clamp_min(1.0), weighted
    )
    rankings.sort(key=lambda item: item["signed_delta"], reverse=True)
    for index, item in enumerate(rankings, start=1):
        item["rank"] = index
    return normalize_signed_attribution(attribution), {
        "objective": objective_id,
        "baseline_score": baseline_score,
        "method_params": {
            "mode": mode,
            "fill_hu": fill_hu,
            "preserve_answer": preserve_answer,
            **spec_options,
        },
        "organs": rankings,
    }
