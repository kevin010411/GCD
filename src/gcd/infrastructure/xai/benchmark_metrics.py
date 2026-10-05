"""Segmentation CAM readouts matching the xai_hw model-space protocol."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm


BEHAVIOUR_METRICS = (
    "gt_dice", "relative_gt_dice", "gt_iou", "prediction_consistency_dice",
    "fixed_roi_class_probability", "relative_fixed_roi_class_probability",
)
SPATIAL_METRICS = (
    "ap_gt", "attribution_mass_in_gt", "dice_at_gt_size", "iou_at_gt_size",
    "mean_boundary_attribution", "mean_interior_attribution",
    "gt_mass_on_boundary_fraction", "gt_mass_in_interior_fraction",
)


def fractions(steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if steps == 21:
        return np.asarray((0, .0005, .001, .0025, .005, .01, .02, .03, .05,
                           .075, .1, .15, .2, .25, .3, .4, .5, .6, .7, .85, 1),
                          dtype=np.float32)
    if steps == 15:
        return np.asarray((0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5,
                           .6, .7, .85, 1), dtype=np.float32)
    return np.linspace(0, 1, steps, dtype=np.float32)


def overlap(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    prediction, reference = np.asarray(prediction, bool), np.asarray(reference, bool)
    intersection = int(np.count_nonzero(prediction & reference))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(reference))
    if denominator == 0:
        return 0.0, 0.0
    return 2 * intersection / denominator, intersection / (denominator - intersection)


def auc(rows: list[dict], metric: str, mode: str) -> float:
    selected = sorted((row for row in rows if row["mode"] == mode),
                      key=lambda row: row["proportion"])
    if len(selected) < 2:
        raise ValueError("AUC needs at least two curve points")
    return float(trapezoid([row[metric] for row in selected],
                           [row["proportion"] for row in selected]))


def spatial_metrics(saliency: np.ndarray, label: np.ndarray,
                    spacing: tuple[float, ...]) -> dict[str, float]:
    scores = np.asarray(saliency, dtype=np.float32).ravel()
    positives = np.asarray(label, dtype=bool).ravel()
    count = int(positives.sum())
    if count == 0 or count == scores.size:
        return {name: float("nan") for name in SPATIAL_METRICS}
    order = np.argsort(-scores, kind="stable")
    top = np.zeros(scores.size, dtype=bool)
    top[order[:count]] = True
    tp = int(np.count_nonzero(top & positives))
    mass = float(scores.sum(dtype=np.float64))
    distances = distance_transform_edt(label, sampling=spacing).ravel()
    boundary = positives & (distances <= 3.0)
    interior = positives & ~boundary
    gt_mass = float(scores[positives].sum(dtype=np.float64))
    return {
        "ap_gt": float(average_precision_score(positives, scores)),
        "attribution_mass_in_gt": gt_mass / mass if mass > 0 else 0.0,
        "dice_at_gt_size": tp / count,
        "iou_at_gt_size": tp / (2 * count - tp),
        "mean_boundary_attribution": float(scores[boundary].mean()) if boundary.any() else 0.0,
        "mean_interior_attribution": float(scores[interior].mean()) if interior.any() else 0.0,
        "gt_mass_on_boundary_fraction": float(scores[boundary].sum(dtype=np.float64) / gt_mass)
        if gt_mass > 0 else 0.0,
        "gt_mass_in_interior_fraction": float(scores[interior].sum(dtype=np.float64) / gt_mass)
        if gt_mass > 0 else 0.0,
    }


def behaviour_curve(*, image: np.ndarray, saliency: np.ndarray, label: np.ndarray,
                    class_id: int, full_prediction: np.ndarray,
                    full_probability: np.ndarray,
                    infer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
                    steps: int = 21, baseline_mode: str = "zero",
                    label_policy: str = "original", modes=("insertion", "deletion"),
                    preserve_answer: bool = False, show_progress: bool = False):
    """One inference per perturbation point, all readouts from that output.

    The optional protected variant keeps all target-class GT voxels at their
    original intensity. It is recorded separately from the standard curve.
    """
    image = np.asarray(image, dtype=np.float32)
    saliency = np.asarray(saliency, dtype=np.float32)
    label = np.asarray(label)
    if image.shape != saliency.shape or image.shape != label.shape:
        raise ValueError("Image, CAM and label must share the model grid")
    if not np.isfinite(image).all():
        raise ValueError("Image must contain finite values")
    if np.shape(full_prediction) != image.shape or np.shape(full_probability) != image.shape:
        raise ValueError("Full prediction/probability must share the model grid")
    if not np.isfinite(saliency).all() or np.any(saliency < 0):
        raise ValueError("CAM must be finite and nonnegative")
    if label_policy not in {"original", "masked"}:
        raise ValueError("label_policy must be original or masked")
    if baseline_mode == "zero":
        baseline = np.zeros_like(image)
    elif baseline_mode == "minimum":
        baseline = np.full_like(image, image.min())
    elif baseline_mode == "mean":
        baseline = np.full_like(image, image.mean())
    else:
        raise ValueError("baseline_mode must be zero, minimum, or mean")
    roi = full_prediction == class_id
    full_dice, _ = overlap(roi, label == class_id)
    full_score = float(full_probability[roi].mean()) if np.any(roi) else float("nan")
    order = np.argsort(-saliency.ravel(), kind="stable")
    protected = (label == class_id).ravel() if preserve_answer else np.zeros(image.size, bool)
    order = order[~protected[order]]
    flat_image, flat_base, flat_label = image.ravel(), baseline.ravel(), label.ravel()
    values = fractions(steps)
    rows = []
    zero = None
    for proportion in tqdm(values, desc="perturbation", disable=not show_progress):
        p = float(proportion)
        selected = order[:int(np.ceil(order.size * p))]
        for mode in modes:
            if mode not in {"insertion", "deletion"}:
                raise ValueError(f"Unknown perturbation mode: {mode}")
            is_full = not preserve_answer and ((mode == "insertion" and p == 1)
                                               or (mode == "deletion" and p == 0))
            is_zero = not preserve_answer and ((mode == "insertion" and p == 0)
                                               or (mode == "deletion" and p == 1))
            if is_full:
                prediction, probability = full_prediction, full_probability
            elif is_zero and zero is not None:
                prediction, probability = zero
            else:
                perturbed = flat_base.copy() if mode == "insertion" else flat_image.copy()
                perturbed[protected] = flat_image[protected]
                perturbed[selected] = flat_image[selected] if mode == "insertion" else flat_base[selected]
                prediction, probability = infer(perturbed.reshape(image.shape))
                if is_zero:
                    zero = prediction, probability
            if label_policy == "original":
                reference = label
            else:
                kept = np.zeros(image.size, bool)
                kept[protected] = True
                kept[selected] = mode == "insertion"
                if mode == "deletion":
                    kept[:] = True
                    kept[selected] = False
                reference = np.where(kept, flat_label, 0).reshape(label.shape)
            gt_dice, gt_iou = overlap(prediction == class_id, reference == class_id)
            if np.shape(prediction) != image.shape or np.shape(probability) != image.shape:
                raise ValueError("Perturbed prediction/probability must share the model grid")
            consistency, _ = overlap(prediction == class_id, roi)
            score = float(probability[roi].mean()) if np.any(roi) else float("nan")
            rows.append({
                "mode": mode, "proportion": p,
                "gt_dice": gt_dice, "relative_gt_dice": gt_dice / full_dice if full_dice > 0 else float("nan"),
                "gt_iou": gt_iou, "prediction_consistency_dice": consistency,
                "fixed_roi_class_probability": score,
                "relative_fixed_roi_class_probability": score / full_score if full_score > 0 else float("nan"),
            })
    return rows, {"full_gt_dice": full_dice,
                  "full_fixed_roi_class_probability": full_score,
                  "full_prediction_voxels": int(roi.sum())}


def evaluate_heatmap(*, image, saliency, label, class_id, full_prediction,
                     full_probability, infer, spacing, steps=21,
                     baseline_mode="zero", label_policy="original",
                     modes=("insertion", "deletion"), preserve_answer=False,
                     show_progress=False):
    """Shared complete single-heatmap evaluation; no CLI or output dependencies."""
    rows, summary = [], {}
    for protected in ((False, True) if preserve_answer else (False,)):
        variant = "preserve_answer" if protected else "standard"
        curve, full = behaviour_curve(
            image=image, saliency=saliency, label=label, class_id=class_id,
            full_prediction=full_prediction, full_probability=full_probability,
            infer=infer, steps=steps, baseline_mode=baseline_mode,
            label_policy=label_policy, modes=modes, preserve_answer=protected,
            show_progress=show_progress,
        )
        rows.extend({"variant": variant, **row} for row in curve)
        summary.update(full)
        for mode in modes:
            for metric in BEHAVIOUR_METRICS:
                summary[f"{variant}_{mode}_{metric}_auc"] = auc(curve, metric, mode)
    summary.update(spatial_metrics(saliency, label == class_id, spacing))
    summary["relative_dice_defined"] = bool(summary["full_gt_dice"] > 0)
    summary["fixed_roi_probability_defined"] = bool(summary["full_prediction_voxels"] > 0)
    return rows, summary
