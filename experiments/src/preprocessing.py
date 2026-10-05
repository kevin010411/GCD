"""Compatibility imports; preprocessing belongs to GCD infrastructure."""
from src.gcd.infrastructure.preprocessing import load_image, preprocess_array, preprocess_tensor

__all__ = ["load_image", "preprocess_array", "preprocess_tensor"]
