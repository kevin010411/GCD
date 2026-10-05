from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=16)
def gaussian_importance_map(size: tuple[int, int, int], sigma_scale=0.125):
    """Score-CAM window weights used for both target prediction and CAM fusion."""
    import torch

    axes = [torch.arange(length, dtype=torch.float32) for length in size]
    grids = torch.meshgrid(*axes, indexing="ij")
    weight = torch.ones(size, dtype=torch.float32)
    scales = (sigma_scale,) * 3 if isinstance(sigma_scale, (int, float)) else sigma_scale
    for grid, length, scale in zip(grids, size, scales):
        sigma = max(float(length) * scale, 1e-6)
        center = (float(length) - 1.0) / 2.0
        weight *= torch.exp(-0.5 * ((grid - center) / sigma) ** 2)
    return (weight / weight.max().clamp_min(1e-8)).clamp_min(1e-3)


def tile_importance_map(size, params, *, default_mode="gaussian"):
    import torch
    mode = params.get("_blend_mode", default_mode)
    if mode == "constant":
        return torch.ones(size, dtype=torch.float32)
    if mode != "gaussian":
        raise ValueError(f"Unsupported tile blend mode: {mode}")
    scale = params.get("_sigma_scale", 0.125)
    if isinstance(scale, list):
        scale = tuple(scale)
    return gaussian_importance_map(size, scale)
