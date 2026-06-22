from __future__ import annotations

import time
from collections.abc import Mapping

from .base import CamPatchContext, XaiLayerSelection, XaiMethod, XaiParameterSpec


def _int_param(params: Mapping[str, object], key: str, default: int) -> int:
    return int(params.get(key, default) or default)


def _float_param(params: Mapping[str, object], key: str, default: float) -> float:
    return float(params.get(key, default) if params.get(key, default) is not None else default)


def _axis_starts(size: int, block_size: int, stride: int) -> list[int]:
    block_size = max(1, min(int(block_size), int(size)))
    stride = max(1, int(stride))
    if block_size >= size:
        return [0]
    starts = list(range(0, size - block_size + 1, stride))
    last = size - block_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _required_model(context: CamPatchContext):
    if context.model is None:
        raise RuntimeError("Perturbation XAI 需要 model 才能重新 forward 遮蔽輸入。")
    return context.model


def _call_objective(objective, logits, target_class: int, reference_mask=None):
    try:
        return objective(logits, target_class, reference_mask)
    except TypeError:
        return objective(logits, target_class)


def _score_batch(logits, target_class: int, objective, reference_mask=None):
    import torch

    scores = []
    for index in range(int(logits.size(0))):
        score = _call_objective(
            objective, logits[index : index + 1], target_class, reference_mask
        )
        scores.append(score.detach().reshape(()).to(logits.device))
    return torch.stack(scores)


def _forward_scores(
    model,
    samples,
    target_class: int,
    objective,
    batch_size: int,
    reference_mask=None,
    params: Mapping[str, object] | None = None,
    preview_regions: list[tuple[slice, slice, slice]] | None = None,
):
    import torch

    params = params or {}
    scores = []
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, int(samples.size(0)), batch_size):
            batch = samples[start : start + batch_size]
            _wait_if_paused(params)
            preview_region = None
            if preview_regions:
                preview_region = preview_regions[start]
            _report_preview(params, batch[0], preview_region)
            logits = model(batch)
            scores.append(_score_batch(logits, target_class, objective, reference_mask))
    return torch.cat(scores)


def _importance_payload(method_id: str, logits, importance_map) -> dict[str, object]:
    return {
        "method": method_id,
        "pred": logits.detach().to("cpu"),
        "importance_map": importance_map.detach().to("cpu"),
    }


def _log_score(params: Mapping[str, object], message: str) -> None:
    logger = params.get("_score_logger")
    if callable(logger):
        logger(message)


def _region_bounds(region: tuple[slice, slice, slice]) -> tuple[int, int, int, int, int, int]:
    return (
        int(region[0].start or 0),
        int(region[0].stop or 0),
        int(region[1].start or 0),
        int(region[1].stop or 0),
        int(region[2].start or 0),
        int(region[2].stop or 0),
    )


def _progress_total(params: Mapping[str, object]) -> int:
    return max(1, int(params.get("_progress_total", 1) or 1))


def _progress_offset(params: Mapping[str, object]) -> int:
    return max(0, int(params.get("_progress_offset", 0) or 0))


def _report_progress(params: Mapping[str, object], done: int) -> None:
    callback = params.get("_progress_callback")
    if not callable(callback):
        return
    total = _progress_total(params)
    current = min(total, _progress_offset(params) + int(done))
    if not _progress_due(params, current, total):
        return
    callback({"current": current, "total": total})


def _progress_due(params: Mapping[str, object], current: int, total: int) -> bool:
    if not isinstance(params, dict):
        return True
    if current >= total:
        return True
    now = time.monotonic()
    min_interval = float(params.get("_progress_min_interval_sec", 0.1) or 0.0)
    last_emit = params.get("_progress_last_emit")
    if last_emit is not None and now - float(last_emit) < min_interval:
        return False
    params["_progress_last_emit"] = now
    return True


def _wait_if_paused(params: Mapping[str, object]) -> None:
    controller = params.get("_pause_controller")
    waiter = getattr(controller, "wait_if_paused", None)
    if callable(waiter):
        waiter()


def _report_preview(
    params: Mapping[str, object],
    sample,
    region: tuple[slice, slice, slice] | None = None,
) -> None:
    callback = params.get("_preview_callback")
    if not callable(callback):
        return
    if not _preview_due(params):
        return
    try:
        data = _preview_sample_data(params, sample, region)
        if data.ndim == 4:
            data = data[0]
        permute = params.get("_preview_permute")
        preview_box = None
        if region is not None:
            preview_box = _preview_box(params, region)
        if permute is not None and data.ndim == 3:
            data = data.permute(*(int(v) for v in permute))
        callback(
            {
                "kind": "perturb_preview",
                "data": data.to("cpu"),
                "spacing": params.get("_preview_spacing"),
                "metadata": params.get("_preview_metadata", {}),
                "preview_box": preview_box,
            }
        )
    except Exception:
        return


def _preview_due(params: Mapping[str, object]) -> bool:
    if not isinstance(params, dict):
        return True
    now = time.monotonic()
    min_interval = float(params.get("_preview_min_interval_sec", 0.2) or 0.0)
    last_emit = params.get("_preview_last_emit")
    if last_emit is not None and now - float(last_emit) < min_interval:
        return False
    params["_preview_last_emit"] = now
    return True


def _preview_sample_data(
    params: Mapping[str, object],
    sample,
    region: tuple[slice, slice, slice] | None,
):
    data = sample.detach()
    full_input = params.get("_preview_full_input")
    if full_input is None or region is None:
        return data
    origin = tuple(int(v) for v in params.get("_preview_tile_origin", (0, 0, 0)))
    full_data = full_input.detach().clone()
    if full_data.ndim == 5:
        full_data = full_data[0]
    if full_data.ndim != 4:
        return data
    tile_data = data[0] if data.ndim == 5 else data
    full_region = _offset_region(region, origin)
    full_data[:, full_region[0], full_region[1], full_region[2]] = tile_data[
        :, region[0], region[1], region[2]
    ]
    return full_data


def _offset_region(
    region: tuple[slice, slice, slice], origin: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(origin[index] + int(axis.start or 0), origin[index] + int(axis.stop or 0))
        for index, axis in enumerate(region)
    )


def _preview_box(
    params: Mapping[str, object], region: tuple[slice, slice, slice]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    origin = tuple(int(v) for v in params.get("_preview_tile_origin", (0, 0, 0)))
    full_region = _offset_region(region, origin)
    starts = [int(axis.start or 0) for axis in full_region]
    ends = [max(int(axis.stop or 0) - 1, starts[index]) for index, axis in enumerate(full_region)]
    permute = params.get("_preview_permute")
    if permute is not None:
        axes = tuple(int(v) for v in permute)
        starts = [starts[axis] for axis in axes]
        ends = [ends[axis] for axis in axes]
    return tuple(starts), tuple(ends)


class _InputPerturbationMethod(XaiMethod):
    family = "perturbation"
    uses_layer_controls = False
    uses_objective = True

    @staticmethod
    def _importance_tensor(patch_payload: dict[str, object], output_size):
        import torch
        import torch.nn.functional as F

        importance_map = patch_payload["importance_map"]
        if not isinstance(importance_map, torch.Tensor):
            raise TypeError("Perturbation payload 缺少 importance_map tensor。")
        if tuple(importance_map.shape[2:]) == tuple(output_size):
            return importance_map
        return F.interpolate(importance_map, size=output_size, mode="trilinear")

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        return self._importance_tensor(patch_payload, selection.output_size)


class PerturbationOcclusionMethod(_InputPerturbationMethod):
    id = "perturb_occlusion"
    display_name = "Occlusion"

    def parameter_schema(self) -> tuple[XaiParameterSpec, ...]:
        return (
            XaiParameterSpec("block_size", "Block Size", "int", default=16, min_value=1, max_value=256, step=1),
            XaiParameterSpec("stride", "Stride", "int", default=8, min_value=1, max_value=256, step=1),
            XaiParameterSpec("baseline", "Baseline", "float", default=0.0, min_value=-10.0, max_value=10.0, step=0.1),
            XaiParameterSpec("batch_size", "Batch Size", "int", default=1, min_value=1, max_value=16, step=1),
        )

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        import torch

        params = context.method_params or {}
        model = _required_model(context)
        block_size = _int_param(params, "block_size", 16)
        stride = _int_param(params, "stride", 8)
        baseline = _float_param(params, "baseline", 0.0)
        batch_size = _int_param(params, "batch_size", 1)
        tile_index = int(params.get("_tile_index", 0) or 0)

        tile = context.input_tensor.detach()
        _, _, depth, height, width = tile.shape
        reference_mask = _reference_mask(context)
        original_score = _call_objective(
            context.objective, context.logits, context.target_class, reference_mask
        ).detach()
        importance = torch.zeros((1, 1, depth, height, width), device=tile.device)
        coverage = torch.zeros_like(importance)

        samples = []
        regions: list[tuple[slice, slice, slice]] = []
        completed = 0
        for z in _axis_starts(depth, block_size, stride):
            for y in _axis_starts(height, block_size, stride):
                for x in _axis_starts(width, block_size, stride):
                    _wait_if_paused(params)
                    zs = slice(z, min(z + block_size, depth))
                    ys = slice(y, min(y + block_size, height))
                    xs = slice(x, min(x + block_size, width))
                    masked = tile.clone()
                    masked[:, :, zs, ys, xs] = baseline
                    samples.append(masked[0])
                    regions.append((zs, ys, xs))

                    if len(samples) == batch_size:
                        stacked = torch.stack(samples, dim=0)
                        scores = _forward_scores(
                            model,
                            stacked,
                            context.target_class,
                            context.objective,
                            batch_size,
                            reference_mask,
                            params,
                            regions,
                        )
                        for score, region in zip(scores, regions):
                            drop = original_score - score
                            zs0, zs1, ys0, ys1, xs0, xs1 = _region_bounds(region)
                            _log_score(
                                params,
                                (
                                    "[perturb_occlusion] "
                                    f"tile={tile_index} "
                                    f"region=z[{zs0}:{zs1}] y[{ys0}:{ys1}] x[{xs0}:{xs1}] "
                                    f"original_score={float(original_score):.6f} "
                                    f"masked_score={float(score):.6f} "
                                    f"drop={float(drop):.6f}"
                                ),
                            )
                            importance[:, :, region[0], region[1], region[2]] += drop
                            coverage[:, :, region[0], region[1], region[2]] += 1
                            completed += 1
                            _report_progress(params, completed)
                        samples = []
                        regions = []

        if samples:
            stacked = torch.stack(samples, dim=0)
            scores = _forward_scores(
                model,
                stacked,
                context.target_class,
                context.objective,
                batch_size,
                reference_mask,
                params,
                regions,
            )
            for score, region in zip(scores, regions):
                drop = original_score - score
                zs0, zs1, ys0, ys1, xs0, xs1 = _region_bounds(region)
                _log_score(
                    params,
                    (
                        "[perturb_occlusion] "
                        f"tile={tile_index} "
                        f"region=z[{zs0}:{zs1}] y[{ys0}:{ys1}] x[{xs0}:{xs1}] "
                        f"original_score={float(original_score):.6f} "
                        f"masked_score={float(score):.6f} "
                        f"drop={float(drop):.6f}"
                    ),
                )
                importance[:, :, region[0], region[1], region[2]] += drop
                coverage[:, :, region[0], region[1], region[2]] += 1
                completed += 1
                _report_progress(params, completed)

        importance = importance / torch.clamp_min(coverage, 1)
        return _importance_payload(self.id, context.logits, importance)


class PerturbationLimeMethod(_InputPerturbationMethod):
    id = "perturb_lime"
    display_name = "LIME"

    def parameter_schema(self) -> tuple[XaiParameterSpec, ...]:
        return (
            XaiParameterSpec("segments_per_axis", "Segments / Axis", "int", default=8, min_value=2, max_value=32, step=1),
            XaiParameterSpec("num_samples", "Samples", "int", default=128, min_value=8, max_value=4096, step=8),
            XaiParameterSpec("kernel_width", "Kernel Width", "float", default=0.25, min_value=0.01, max_value=10.0, step=0.05),
            XaiParameterSpec("baseline", "Baseline", "float", default=0.0, min_value=-10.0, max_value=10.0, step=0.1),
            XaiParameterSpec("batch_size", "Batch Size", "int", default=1, min_value=1, max_value=16, step=1),
            XaiParameterSpec("random_seed", "Random Seed", "int", default=0, min_value=0, max_value=999999, step=1),
        )

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        import numpy as np
        import torch

        params = context.method_params or {}
        model = _required_model(context)
        segments_per_axis = _int_param(params, "segments_per_axis", 8)
        num_samples = _int_param(params, "num_samples", 128)
        kernel_width = max(0.01, _float_param(params, "kernel_width", 0.25))
        baseline = _float_param(params, "baseline", 0.0)
        batch_size = _int_param(params, "batch_size", 1)
        random_seed = _int_param(params, "random_seed", 0)

        tile = context.input_tensor.detach()
        _, _, depth, height, width = tile.shape
        labels = _regular_grid_labels(
            depth, height, width, segments_per_axis, tile.device
        )
        reference_mask = _reference_mask(context)
        feature_count = int(labels.max().item()) + 1
        rng = np.random.default_rng(random_seed)
        samples_np = rng.integers(0, 2, size=(num_samples, feature_count), dtype=np.int64)
        samples_np[0, :] = 1

        scores = []
        completed = 0
        for start in range(0, num_samples, batch_size):
            _wait_if_paused(params)
            current = min(batch_size, num_samples - start)
            batch_masks = torch.as_tensor(
                samples_np[start : start + batch_size],
                device=tile.device,
                dtype=torch.float32,
            )
            spatial_masks = _feature_masks_to_volume(batch_masks, labels)
            masked = tile * spatial_masks + baseline * (1.0 - spatial_masks)
            scores.append(
                _forward_scores(
                    model,
                    masked,
                    context.target_class,
                    context.objective,
                    batch_size,
                    reference_mask,
                    params,
                )
            )
            completed += current
            _report_progress(params, completed)
        y = torch.cat(scores).detach().cpu().numpy()

        distances = _cosine_distances_to_full(samples_np)
        weights = np.exp(-((distances**2) / (kernel_width**2)))
        coefficients = _weighted_ridge(samples_np.astype(np.float64), y, weights)
        coef_tensor = torch.as_tensor(coefficients, device=tile.device, dtype=torch.float32)
        importance = coef_tensor[labels].unsqueeze(0).unsqueeze(0)
        return _importance_payload(self.id, context.logits, importance)


class PerturbationRiseMethod(_InputPerturbationMethod):
    id = "perturb_rise"
    display_name = "RISE"

    def parameter_schema(self) -> tuple[XaiParameterSpec, ...]:
        return (
            XaiParameterSpec("num_masks", "Masks", "int", default=64, min_value=8, max_value=4096, step=8),
            XaiParameterSpec("mask_grid_size", "Mask Grid", "int", default=8, min_value=2, max_value=64, step=1),
            XaiParameterSpec("keep_probability", "Keep Probability", "float", default=0.5, min_value=0.05, max_value=0.95, step=0.05),
            XaiParameterSpec("baseline", "Baseline", "float", default=0.0, min_value=-10.0, max_value=10.0, step=0.1),
            XaiParameterSpec("batch_size", "Batch Size", "int", default=1, min_value=1, max_value=16, step=1),
            XaiParameterSpec("random_seed", "Random Seed", "int", default=0, min_value=0, max_value=999999, step=1),
        )

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        import torch
        import torch.nn.functional as F

        params = context.method_params or {}
        model = _required_model(context)
        num_masks = _int_param(params, "num_masks", 64)
        grid_size = _int_param(params, "mask_grid_size", 8)
        keep_probability = min(0.95, max(0.05, _float_param(params, "keep_probability", 0.5)))
        baseline = _float_param(params, "baseline", 0.0)
        batch_size = _int_param(params, "batch_size", 1)
        random_seed = _int_param(params, "random_seed", 0)

        tile = context.input_tensor.detach()
        _, _, depth, height, width = tile.shape
        generator = torch.Generator(device=tile.device)
        generator.manual_seed(random_seed)
        reference_mask = _reference_mask(context)
        original_score = _call_objective(
            context.objective, context.logits, context.target_class, reference_mask
        ).detach()
        importance = torch.zeros((1, 1, depth, height, width), device=tile.device)
        exposure = torch.zeros_like(importance)

        for start in range(0, num_masks, batch_size):
            _wait_if_paused(params)
            current = min(batch_size, num_masks - start)
            low_res = (
                torch.rand(
                    (current, 1, grid_size, grid_size, grid_size),
                    generator=generator,
                    device=tile.device,
                )
                < keep_probability
            ).to(torch.float32)
            masks = F.interpolate(
                low_res, size=(depth, height, width), mode="trilinear", align_corners=False
            )
            masked = tile * masks + baseline * (1.0 - masks)
            scores = _forward_scores(
                model,
                masked,
                context.target_class,
                context.objective,
                batch_size,
                reference_mask,
                params,
            )
            drops = (original_score - scores).reshape(current, 1, 1, 1, 1)
            importance += torch.sum(drops * masks, dim=0, keepdim=True)
            exposure += torch.sum(masks, dim=0, keepdim=True)
            _report_progress(params, start + current)

        importance = importance / torch.clamp_min(exposure, 1e-6)
        return _importance_payload(self.id, context.logits, importance)


def _regular_grid_labels(depth: int, height: int, width: int, segments_per_axis: int, device):
    import torch

    segments = max(2, int(segments_per_axis))
    z_edges = torch.linspace(0, depth, segments + 1, device=device).round().to(torch.int64)
    y_edges = torch.linspace(0, height, segments + 1, device=device).round().to(torch.int64)
    x_edges = torch.linspace(0, width, segments + 1, device=device).round().to(torch.int64)
    labels = torch.empty((depth, height, width), dtype=torch.long, device=device)
    index = 0
    for zi in range(segments):
        for yi in range(segments):
            for xi in range(segments):
                zs = slice(int(z_edges[zi]), int(z_edges[zi + 1]))
                ys = slice(int(y_edges[yi]), int(y_edges[yi + 1]))
                xs = slice(int(x_edges[xi]), int(x_edges[xi + 1]))
                if zs.start == zs.stop or ys.start == ys.stop or xs.start == xs.stop:
                    continue
                labels[zs, ys, xs] = index
                index += 1
    return labels


def _reference_mask(context: CamPatchContext):
    import torch

    params = context.method_params or {}
    reference_mask = params.get("reference_mask")
    if isinstance(reference_mask, torch.Tensor):
        return reference_mask.to(device=context.logits.device, dtype=torch.bool)
    return _prediction_mask(context.logits, context.target_class)


def _prediction_mask(logits, target_class: int):
    import torch

    return torch.argmax(logits[0], dim=0) == target_class


def _feature_masks_to_volume(feature_masks, labels):
    return feature_masks[:, labels].unsqueeze(1)


def _cosine_distances_to_full(samples):
    import numpy as np

    feature_count = samples.shape[1]
    active = np.sum(samples, axis=1)
    norm = np.sqrt(np.maximum(active, 1) * feature_count)
    similarity = active / norm
    return 1.0 - similarity


def _weighted_ridge(samples, scores, weights):
    try:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(samples, scores, sample_weight=weights)
        return model.coef_
    except Exception:
        import numpy as np

        x = np.asarray(samples, dtype=np.float64)
        y = np.asarray(scores, dtype=np.float64)
        w = np.sqrt(np.asarray(weights, dtype=np.float64)).reshape(-1, 1)
        x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        xw = x_aug * w
        yw = y * w.reshape(-1)
        ridge = np.eye(x_aug.shape[1], dtype=np.float64)
        ridge[0, 0] = 0.0
        beta = np.linalg.pinv(xw.T @ xw + ridge) @ xw.T @ yw
        return beta[1:]
