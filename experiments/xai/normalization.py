from __future__ import annotations


def normalize_attribution(attribution):
    """Normalize one 3-D attribution map to [0, 1]."""
    import torch

    attribution = torch.nan_to_num(
        attribution.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    attribution = attribution - attribution.min()
    maximum = attribution.max()
    return (
        attribution / maximum
        if float(maximum) > 0.0
        else torch.zeros_like(attribution)
    )


def normalize_signed_attribution(attribution):
    """Normalize a signed attribution map to [-1, 1] without losing direction."""
    import torch

    attribution = torch.nan_to_num(
        attribution.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    maximum = attribution.abs().max()
    return (
        attribution / maximum
        if float(maximum) > 0.0
        else torch.zeros_like(attribution)
    )


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
