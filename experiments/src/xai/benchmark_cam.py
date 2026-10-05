"""Compatibility imports; CAM algorithms belong to GCD src."""
from src.gcd.infrastructure.xai.methods.benchmark_cam import (
    FAMILIES, LayerTap, feature_layers, window_slices, gaussian_importance,
    patch_cam, normalize, generate,
)

__all__ = ["FAMILIES", "LayerTap", "feature_layers", "window_slices",
           "gaussian_importance", "patch_cam", "normalize", "generate"]
