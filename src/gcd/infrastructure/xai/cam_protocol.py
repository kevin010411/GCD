"""Validated CAM protocol shared by UI and benchmark (no Qt dependency)."""

DEFAULT_CAM_PROTOCOL = dict(target_region="full_prediction", reduction="mean", relu_stage="per_tile")


def resolve_cam_protocol(value=None):
    value = dict(value or {})
    unknown = set(value) - set(DEFAULT_CAM_PROTOCOL)
    if unknown:
        raise ValueError(f"Unknown cam_protocol keys: {sorted(unknown)}")
    result = {**DEFAULT_CAM_PROTOCOL, **value}
    for key, choices in dict(target_region=("full_prediction", "tile_prediction"),
                             reduction=("mean", "sum"),
                             relu_stage=("per_tile", "after_fusion")).items():
        if result[key] not in choices:
            raise ValueError(f"cam_protocol.{key} must be one of {choices}")
    return result


def target_score(logits, class_id, mask, reduction="mean"):
    values = logits[:, class_id]
    mask = mask.to(device=values.device).bool().clone()
    if mask.shape != values.shape:
        raise ValueError("CAM target mask and logits grids differ")
    selected = values[mask]
    if not selected.numel():
        return values.sum() * 0
    return selected.mean() if reduction == "mean" else selected.sum()
