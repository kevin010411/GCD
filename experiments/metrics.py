from __future__ import annotations

from typing import Any


def segmentation_metrics(
    prediction, target, class_count: int, cfg: Any
) -> dict[str, Any]:
    """Calculate per-class and macro Dice/IoU for segmentation tensors."""
    import torch

    prediction = prediction.to(torch.int64)
    target = target.to(torch.int64)
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            "Prediction and ground truth shapes differ after resampling: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )

    start = 0 if bool(cfg.metrics.include_background) else 1
    empty_score = float(cfg.metrics.empty_score)
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id in range(start, class_count):
        pred_mask = prediction == class_id
        target_mask = target == class_id
        intersection = int(torch.logical_and(pred_mask, target_mask).sum())
        pred_count = int(pred_mask.sum())
        target_count = int(target_mask.sum())
        union = pred_count + target_count - intersection
        denominator = pred_count + target_count
        dice = empty_score if denominator == 0 else 2 * intersection / denominator
        iou = empty_score if union == 0 else intersection / union
        per_class[str(class_id)] = {
            "dice": dice,
            "iou": iou,
            "predicted_voxels": pred_count,
            "target_voxels": target_count,
        }

    dice_values = [float(value["dice"]) for value in per_class.values()]
    iou_values = [float(value["iou"]) for value in per_class.values()]
    return {
        "per_class": per_class,
        "mean_dice": sum(dice_values) / len(dice_values) if dice_values else None,
        "mean_iou": sum(iou_values) / len(iou_values) if iou_values else None,
    }
