"""Shared model-space tiled CAM core for UI and batch benchmarks.

The default matches xai_hw's raw patch CAM protocol; cam_protocol can select
tile-local targets, summed logits and post-fusion rectification instead.
Normalize once over the complete volume. This module accepts GCD models with 3-D
feature layers and does not depend on UI display resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from ..cam_protocol import resolve_cam_protocol, target_score


FAMILIES = ("gradcam", "hirescam", "layercam", "scorecam")


@dataclass(frozen=True)
class LayerTap:
    module: torch.nn.Module
    capture_input: bool = False


def feature_layers(model: torch.nn.Module, names: tuple[str, ...] | None = None):
    """Resolve exact module paths or the xai_hw last-three-Conv3d convention."""
    if names:
        targets = getattr(model, "xai_layer_targets", {})
        layers = []
        for name in names:
            base, suffix = (name[:-6], ":input") if name.endswith(":input") else (name, "")
            module = model.get_submodule(targets.get(base, base))
            layers.append((name, LayerTap(module, bool(suffix))))
    else:
        convolutions = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, torch.nn.Conv3d)]
        if len(convolutions) < 4:
            raise ValueError("Auto layer selection requires at least four Conv3d modules")
        layers = [(name, LayerTap(module)) for name, module in convolutions[-4:-1]]
    if len(layers) != 3 or len({(id(tap.module), tap.capture_input) for _, tap in layers}) != 3:
        raise ValueError("Specify three distinct feature layers in L1, L2, L3 order")
    return layers


def window_slices(shape: tuple[int, ...], roi: tuple[int, ...], overlap: float):
    starts = []
    for length, size in zip(shape, roi):
        if length < size:
            raise ValueError(f"Pad model input to ROI first: {shape} versus {roi}")
        stride = max(1, int(size * (1.0 - overlap)))
        positions = list(range(0, length - size + 1, stride))
        if positions[-1] != length - size:
            positions.append(length - size)
        starts.append(positions)
    for origin in product(*starts):
        yield tuple(slice(start, start + size) for start, size in zip(origin, roi))


def gaussian_importance(roi: tuple[int, ...], device: torch.device, sigma_scale=0.125):
    axes = [torch.arange(size, device=device, dtype=torch.float32) for size in roi]
    grids = torch.meshgrid(*axes, indexing="ij")
    result = torch.ones(roi, device=device, dtype=torch.float32)
    scales = (sigma_scale,) * len(roi) if isinstance(sigma_scale, (int, float)) else sigma_scale
    for grid, size, scale in zip(grids, roi, scales):
        center = (size - 1.0) / 2.0
        result *= torch.exp(-0.5 * ((grid - center) / (scale * size)) ** 2)
    return (result / result.max()).clamp_min(1e-3)


def _logits(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 5:
        raise ValueError("CAM requires logits shaped [batch, class, x, y, z]")
    return output


def _capture(model, layer, window, *, gradient: bool, class_id: int, target, reduction="mean"):
    captured = {}
    gradient_handles = []

    def hook(_module, _inputs, output):
        activation = _inputs[0] if layer.capture_input else output
        if not isinstance(activation, torch.Tensor):
            raise ValueError("Selected feature layer must output a tensor")
        captured["activation"] = activation
        if gradient:
            gradient_handles.append(activation.register_hook(
                lambda value: captured.setdefault("gradient", value)))

    handle = layer.module.register_forward_hook(hook)
    try:
        if gradient:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = _logits(model(window))
                target_score(logits, class_id, target[None], reduction).backward()
            return captured["activation"].detach(), captured["gradient"].detach()
        with torch.inference_mode():
            _logits(model(window))
        return captured["activation"].detach(), None
    finally:
        handle.remove()
        # The tensor hook closes over captured, which itself owns the activation.
        # Remove it after backward to break that cycle and release each tile promptly.
        for gradient_handle in gradient_handles:
            gradient_handle.remove()
        if gradient:
            model.zero_grad(set_to_none=True)


def patch_cam(model, layer, window, class_id: int, target, family: str,
              score_batch_size: int = 16, cam_protocol=None):
    protocol = resolve_cam_protocol(cam_protocol)
    if not bool(target.any()):
        return torch.zeros(window.shape[-3:], device=window.device)
    activation, gradient = _capture(model, layer, window, gradient=family != "scorecam",
                                    class_id=class_id, target=target, reduction=protocol["reduction"])
    if family in FAMILIES[:3]:
        cam = raw_gradient_cam(activation, gradient, family, rectify=protocol["relu_stage"] == "per_tile")[0, 0]
    elif family == "scorecam":
        activated = activation[0].relu()
        resized = F.interpolate(activated[None], size=window.shape[-3:],
                                mode="trilinear", align_corners=False)[0]
        flat = resized.flatten(start_dim=1)
        minimum = flat.min(dim=1).values[:, None, None, None]
        maximum = flat.max(dim=1).values[:, None, None, None]
        masks = (resized - minimum) / (maximum - minimum).clamp_min(1e-8)
        target_float = target.to(dtype=window.dtype)
        count = target_float.sum().clamp_min(1.0)
        scores = []
        with torch.inference_mode():
            for masked in (window * masks[:, None]).split(score_batch_size):
                logits = _logits(model(masked))
                scores.append((logits[:, class_id] * target_float).sum(dim=(-3, -2, -1)) /
                              (count if protocol["reduction"] == "mean" else 1.0))
        weights = torch.softmax(torch.cat(scores), dim=0)
        cam = (weights[:, None, None, None] * activated).sum(dim=0).relu()
    else:
        raise ValueError(f"Unknown CAM family: {family}")
    return F.interpolate(cam[None, None], size=window.shape[-3:],
                         mode="trilinear", align_corners=False)[0, 0].detach()


def raw_gradient_cam(activation, gradient, family, rectify=True):
    """Unnormalized CAM, optionally signed; preserve magnitude before fusion."""
    if family == "gradcam":
        value = (activation * gradient.mean(dim=(-3, -2, -1), keepdim=True)).sum(dim=1, keepdim=True)
    elif family == "hirescam":
        value = (activation * gradient).sum(dim=1, keepdim=True)
    elif family == "layercam":
        value = (activation * gradient.relu()).sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown gradient CAM: {family}")
    return value.relu() if rectify else value


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    if high - low <= np.finfo(np.float32).eps:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def generate(model, image: torch.Tensor, full_logits: torch.Tensor, class_id: int,
             method: str, layers, roi: tuple[int, ...], overlap: float,
             score_batch_size: int = 16, spacing: tuple[float, ...] = (1., 1., 1.),
             blend_mode: str = "gaussian", sigma_scale=0.125, cam_protocol=None):
    """Return a normalized map in the exact model input grid."""
    if method == "prediction_logits_minmax":
        return normalize(full_logits[0, class_id].detach().cpu().numpy())
    protocol = resolve_cam_protocol(cam_protocol)
    prediction = full_logits.argmax(dim=1)[0]
    if method == "prediction_mask_binary":
        return (prediction == class_id).cpu().numpy().astype(np.float32)
    if method == "prediction_distance_prior":
        from scipy.ndimage import distance_transform_edt
        return normalize(distance_transform_edt((prediction == class_id).cpu().numpy(),
                                                sampling=spacing))
    if method.startswith("multilayer_"):
        family = method.removeprefix("multilayer_")
        if family not in FAMILIES[:3]:
            raise ValueError(f"Unsupported multi-layer method: {method}")
        selected = [layer for _, layer in layers]
    else:
        family, suffix = method.rsplit("_L", 1)
        if family not in FAMILIES or suffix not in {"1", "2", "3"}:
            raise ValueError(f"Unknown method: {method}")
        selected = [layers[int(suffix) - 1][1]]
    shape = tuple(image.shape[-3:])
    accum = torch.zeros(shape, device=image.device, dtype=torch.float32)
    coverage = torch.zeros_like(accum)
    if blend_mode not in {"constant", "gaussian"}:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")
    importance = (gaussian_importance(roi, image.device, sigma_scale)
                  if blend_mode == "gaussian" else torch.ones(roi, device=image.device))
    for spatial in window_slices(shape, roi, overlap):
        window = image[(slice(None), slice(None), *spatial)]
        target = prediction[spatial] == class_id
        if protocol["target_region"] == "tile_prediction":
            with torch.inference_mode():
                target = _logits(model(window)).argmax(dim=1)[0] == class_id
        local = torch.stack([patch_cam(model, layer, window, class_id, target, family,
                                       score_batch_size, protocol) for layer in selected]).mean(dim=0)
        accum[spatial] += local.float() * importance
        coverage[spatial] += importance
    return normalize((accum / coverage.clamp_min(1e-8)).relu().detach().cpu().numpy())
