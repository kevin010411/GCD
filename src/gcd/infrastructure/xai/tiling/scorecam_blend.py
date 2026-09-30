from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=16)
def gaussian_importance_map(size: tuple[int, int, int]):
    """Score-CAM window weights used for both target prediction and CAM fusion."""
    import torch

    axes = [torch.arange(length, dtype=torch.float32) for length in size]
    grids = torch.meshgrid(*axes, indexing="ij")
    weight = torch.ones(size, dtype=torch.float32)
    for grid, length in zip(grids, size):
        sigma = max(float(length) * 0.125, 1e-6)
        center = (float(length) - 1.0) / 2.0
        weight *= torch.exp(-0.5 * ((grid - center) / sigma) ** 2)
    return (weight / weight.max().clamp_min(1e-8)).clamp_min(1e-3)
