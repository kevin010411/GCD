from __future__ import annotations

from contextlib import nullcontext
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

    legacy_xai = cfg.get("xai", {})
    params = dict(
        method_params
        if method_params is not None
        else legacy_xai.get("method_params", {}).get(method_id, {})
    )
    requested_layer = str(
        layer
        if layer is not None and layer
        else legacy_xai.get("layer", "") or cfg.get("default_layer", "")
    )
    if xai_method.uses_layer_controls and requested_layer:
        params["_selected_layer"] = requested_layer
    if method_id == "scorecam":
        params["_objective_id"] = "predicted_target_mask"
        if params.pop("_ui_tiled", False):
            return _compute_ui_tiled_scorecam(
                batch=batch,
                model=model,
                method=xai_method,
                target_class=target_class,
                cfg=cfg,
                layer=requested_layer,
                params=params,
                return_metadata=return_metadata,
            )
    volume_shape = tuple(batch.shape[2:])
    roi_size = tuple(int(value) for value in cfg.inference.roi_size)
    sample = F.interpolate(
        batch.detach(), size=roi_size, mode="trilinear", align_corners=False
    )
    if xai_method.family == "gradient" and method_id != "scorecam":
        sample.requires_grad_(True)
    hook_manager = (
        XaiLayerHookManager(
            model,
            selected_layers=(requested_layer,) if requested_layer else None,
        )
        if xai_method.uses_layer_controls else None
    )

    model.zero_grad(set_to_none=True)
    forward_context = torch.no_grad() if method_id == "scorecam" else nullcontext()
    if hook_manager is None:
        with forward_context:
            logits = predictor(sample)
        layers_by_name = {}
    else:
        with hook_manager:
            with forward_context:
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
    metadata = {}
    if method_id == "scorecam":
        weights = payload["weights"]
        valid = payload["valid_channels"]
        valid_count = int(valid.sum())
        selected_scores = payload["scores"][valid]
        metadata["scorecam_diagnostics"] = {
            "pipeline": "single_resized_roi",
            "layer": selected_layer,
            "feature_count": channel_count,
            "activation_shape": list(layers_by_name[selected_layer].shape),
            "valid_features": valid_count,
            "masked_forwards": int(payload["masked_forward_count"]),
            "original_target_voxels": int((logits.argmax(dim=1) == target_class).sum()),
            "score_span": (
                float(selected_scores.max() - selected_scores.min())
                if valid_count else 0.0
            ),
            "max_weight": float(weights.max()) if valid_count else 0.0,
            "effective_features": (
                1.0 / float(weights.square().sum()) if valid_count else 0.0
            ),
            "heatmap_positive_fraction": float((attribution > 0).float().mean()),
            "heatmap_above_0_5_fraction": float((attribution >= 0.5).float().mean()),
            "heatmap_above_0_8_fraction": float((attribution >= 0.8).float().mean()),
        }
    return (attribution, metadata) if return_metadata else attribution


def _compute_ui_tiled_scorecam(
    *, batch, model, method, target_class, cfg, layer, params, return_metadata
):
    """Run the GUI Score-CAM tile collection and aggregation on experiment input."""
    from tqdm.auto import tqdm

    from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
    from src.gcd.infrastructure.xai.runners.xai_cam_runner import (
        XaiCamRunRequest,
        XaiCamRunner,
    )
    from src.gcd.infrastructure.xai.tiling.tile_collector import (
        TileCollectionRequest,
        TileCollector,
    )
    from src.gcd.infrastructure.xai.tiling.tile_strategy import SlidingWindowTileStrategy

    if batch.ndim != 5 or batch.shape[0] != 1 or batch.shape[1] != 1:
        raise ValueError("UI-tiled Score-CAM expects one single-channel 3-D volume.")
    size = int(cfg.get("size", cfg.inference.roi_size[0]))
    stride = min(int(cfg.get("stride", size)), max(1, size * 3 // 4))
    plan = SlidingWindowTileStrategy().plan(
        input_shape=tuple(int(v) for v in batch.shape[2:]),
        patch_size=size,
        stride=stride,
    )
    params = {**params, "_selected_layer": layer, "_feature_start": 0, "_feature_stop": 999}
    with tqdm(total=len(plan.regions), desc="Score-CAM tiles", unit="tile") as progress:
        def update_progress(_message):
            progress.update(1)

        collection = TileCollector().collect(TileCollectionRequest(
            model=model,
            device=batch.device,
            model_input=batch[0].detach(),
            method=method,
            objective=GradCamEngine._predicted_target_mask_objective,
            target_class=int(target_class),
            tile_plan=plan,
            method_params=params,
            logger=update_progress,
        ))
    result = XaiCamRunner().run(XaiCamRunRequest(
        method=method,
        patches=collection.patches,
        layers=collection.layers,
        img1=batch[0].detach().cpu(),
        size=size,
        stride=stride,
        permute=(0, 1, 2),
        default_layer=layer,
        tile_plan=plan,
        layer=layer,
        n1=0,
        n2=999,
    ))
    attribution = result.cam
    patches = collection.patches
    effective = [
        1.0 / float(patch["weights"].square().sum())
        for patch in patches if int(patch["masked_forward_count"]) > 0
    ]
    metadata = {"scorecam_diagnostics": {
        "pipeline": "ui_tiled",
        "layer": layer,
        "feature_count": int(collection.layers[layer]),
        "tile_count": len(patches),
        "global_prediction_forwards": (
            len(patches) if params.get("_objective_id") == "predicted_target_mask" else 0
        ),
        "target_mask_source": (
            "full_volume_prediction"
            if params.get("_objective_id") == "predicted_target_mask"
            else "objective_specific"
        ),
        "tile_blend": "gaussian_sigma_0.125",
        "activation": "relu_before_mask_and_weighting",
        "masked_forwards": sum(int(patch["masked_forward_count"]) for patch in patches),
        "zero_target_tiles": sum(not bool(patch["target_present"]) for patch in patches),
        "effective_features_mean": sum(effective) / len(effective) if effective else 0.0,
        "heatmap_positive_fraction": float((attribution > 0).float().mean()),
        "heatmap_above_0_5_fraction": float((attribution >= 0.5).float().mean()),
        "heatmap_above_0_8_fraction": float((attribution >= 0.8).float().mean()),
        "ui_target_voxels": int((result.model_output == target_class).sum()),
    }}
    del patches, collection
    return (attribution, metadata) if return_metadata else attribution


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
