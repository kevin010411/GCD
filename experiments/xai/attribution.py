from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .normalization import normalize_attribution
from .organ_occlusion import (
    SUPPORTED_DATASET_OBJECTIVES,
    compute_organ_occlusion_attribution,
)


def compute_attribution(
    batch,
    model,
    predictor: Callable[[Any], Any],
    method: str,
    target_class: int,
    target_mask,
    cfg: Any,
    *,
    dataset_context: dict[str, Any] | None = None,
    return_metadata: bool = False,
    method_params: dict[str, Any] | None = None,
    layer: str | None = None,
    objective=None,
    xai_method_override=None,
):
    """Compute a 3-D map with the same XAI registry/methods used by the GUI."""
    import torch
    import torch.nn.functional as F
    from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
    from src.gcd.infrastructure.xai.methods import (
        CamPatchContext,
        XaiLayerSelection,
    )
    from src.gcd.infrastructure.xai.methods.registry import XaiMethodRegistry
    from src.gcd.infrastructure.xai.runtime.layer_hooks import XaiLayerHookManager

    method_id = str(method).lower()
    cam_objective = GradCamEngine._predicted_target_mask_objective
    xai_method = _resolve_method(
        method_id, cam_objective, xai_method_override, XaiMethodRegistry
    )
    if xai_method.execution_scope == "dataset":
        return _compute_legacy_dataset_attribution(
            batch=batch,
            method_id=method_id,
            target_class=target_class,
            cfg=cfg,
            dataset_context=dataset_context,
            return_metadata=return_metadata,
            method_params=method_params,
            objective=objective,
        )

    volume_shape = tuple(batch.shape[2:])
    roi_size = tuple(int(value) for value in cfg.inference.roi_size)
    sample = F.interpolate(
        batch.detach(), size=roi_size, mode="trilinear", align_corners=False
    )
    if xai_method.family == "gradient":
        sample.requires_grad_(True)
    legacy_xai = cfg.get("xai", {})
    params = dict(
        method_params
        if method_params is not None
        else legacy_xai.get("method_params", {}).get(method_id, {})
    )
    hook_manager = (
        XaiLayerHookManager(model) if xai_method.uses_layer_controls else None
    )

    model.zero_grad(set_to_none=True)
    if hook_manager is None:
        logits = predictor(sample)
        layers_by_name = {}
    else:
        with hook_manager:
            logits = predictor(sample)
            layers_by_name = hook_manager.layers_by_name()
            payload = xai_method.collect_patch_data(
                _patch_context(
                    CamPatchContext,
                    sample,
                    logits,
                    layers_by_name,
                    target_class,
                    cam_objective,
                    model,
                    params,
                    batch.device,
                )
            )
    if hook_manager is None:
        payload = xai_method.collect_patch_data(
            _patch_context(
                CamPatchContext,
                sample,
                logits,
                layers_by_name,
                target_class,
                cam_objective,
                model,
                params,
                batch.device,
            )
        )

    if xai_method.uses_layer_controls:
        requested_layer = str(
            layer
            if layer is not None
            else legacy_xai.get("layer", "") or cfg.get("default_layer", "")
        )
        selected_layer = (
            requested_layer
            if requested_layer in layers_by_name
            else next(iter(layers_by_name))
        )
        channel_count = int(layers_by_name[selected_layer].shape[1])
    else:
        selected_layer = "input"
        channel_count = 1
    attribution = xai_method.build_tile_cam(
        payload,
        XaiLayerSelection(selected_layer, 0, channel_count, roi_size),
        method_params=params,
    )
    if xai_method.family == "gradient":
        attribution = torch.relu(attribution)
    attribution = F.interpolate(
        attribution, size=volume_shape, mode="trilinear", align_corners=False
    )[0, 0]
    attribution = normalize_attribution(attribution.detach())
    return (attribution, {}) if return_metadata else attribution


def _resolve_method(method_id, objective, override, registry_type):
    if override is not None:
        return override
    registry = registry_type.default(objective)
    available = registry.methods_by_id
    if method_id not in available:
        raise ValueError(
            f"Unknown GUI XAI method {method_id!r}; choose {sorted(available)}"
        )
    return registry.resolve(method_id)


def _patch_context(
    context_type,
    sample,
    logits,
    layers_by_name,
    target_class,
    objective,
    model,
    method_params,
    device,
):
    return context_type(
        input_tensor=sample,
        logits=logits,
        layers_by_name=layers_by_name,
        target_class=target_class,
        objective=objective,
        model=model,
        method_params=method_params,
        device=device,
    )


def _compute_legacy_dataset_attribution(
    *,
    batch,
    method_id: str,
    target_class: int,
    cfg: Any,
    dataset_context: dict[str, Any] | None,
    return_metadata: bool,
    method_params: dict[str, Any] | None,
    objective,
):
    if dataset_context is None:
        raise ValueError(
            f"XAI method {method_id!r} requires dataset_context in experiments CLI"
        )
    from src.gcd.infrastructure.organ_occlusion import TotalSegmentatorOrganService

    legacy_xai = cfg.get("xai", {})
    params = dict(
        method_params
        if method_params is not None
        else legacy_xai.get("method_params", {}).get(method_id, {})
    )
    objective_id = (
        str(getattr(objective, "id", ""))
        if objective is not None
        else str(legacy_xai.get("objective", "predicted_mask_dice"))
    )
    if objective is None and objective_id not in SUPPORTED_DATASET_OBJECTIVES:
        raise ValueError(
            f"Unsupported xai.objective {objective_id!r}; choose "
            f"{sorted(SUPPORTED_DATASET_OBJECTIVES)}"
        )
    cache_root = params.pop("cache_root", "output/organ_masks")
    service = TotalSegmentatorOrganService(cache_root)
    device_name = "gpu" if str(batch.device).startswith("cuda") else "cpu"
    organ_masks = service.run(
        dataset_context["input_path"],
        device=device_name,
        task=str(params.pop("task", "total")),
        merge_organs=bool(params.pop("merge_organs", True)),
    )
    progress_start = dataset_context.get("progress_start_callback")
    if progress_start is not None:
        progress_start(len(organ_masks) + 1)
    progress_callback = dataset_context.get("progress_callback")
    if progress_callback is not None:
        progress_callback(1)
    attribution, metadata = compute_organ_occlusion_attribution(
        source=dataset_context["source"],
        affine=dataset_context["affine"],
        organ_masks=organ_masks,
        preprocess_image=dataset_context["preprocess_image"],
        infer=dataset_context["infer"],
        baseline_logits=dataset_context["baseline_logits"],
        baseline_input=batch,
        answer_mask=dataset_context.get(
            "answer_mask",
            dataset_context["baseline_logits"].argmax(dim=1)[0] == int(target_class),
        ),
        target_class=target_class,
        objective_id=objective_id,
        method_params=params,
        device=batch.device,
        progress_callback=progress_callback,
        objective=objective,
    )
    metadata["totalsegmentator"] = service.last_run_metadata
    return (attribution, metadata) if return_metadata else attribution
