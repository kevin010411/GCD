from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..methods.cam_methods import CamMethod, CamPatchContext
from ..runtime.layer_hooks import XaiLayerHookManager
from .tile_strategy import TilePlan, TileRegion

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TileCollectionRequest:
    model: Any
    device: Any
    model_input: torch.Tensor
    method: CamMethod
    objective: Callable[[torch.Tensor, int], torch.Tensor]
    target_class: int
    tile_plan: TilePlan
    method_params: Mapping[str, object] | None = None
    model_input_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    model_input_metadata: Mapping[str, object] | None = None
    display_permute: tuple[int, int, int] = (0, 1, 2)
    reference_mask: torch.Tensor | None = None
    progress_units: Callable[[str, Mapping[str, object]], int] | None = None
    pause_waiter: Callable[[CamMethod, Mapping[str, object]], None] | None = None
    logger: Callable[[str], None] | None = None


@dataclass(frozen=True)
class TileCollectionResult:
    patches: list[dict[str, object]]
    layers: dict[str, int]
    model_output: torch.Tensor
    tile_plan: TilePlan


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None


class TileCollector:
    def collect(self, request: TileCollectionRequest) -> TileCollectionResult:
        import torch

        method_params = dict(request.method_params or {})
        img2 = request.model_input.unsqueeze(0).to(request.device)
        if request.method.family == "gradient":
            img2.requires_grad_()

        hook_manager = (
            XaiLayerHookManager(request.model)
            if request.method.uses_layer_controls
            else None
        )
        patches: list[dict[str, object]] = []
        logits = None
        progress_units = (
            request.progress_units(request.method.id, method_params)
            if request.progress_units is not None
            else 1
        )
        progress_total = progress_units * len(request.tile_plan.regions)
        progress_callback = method_params.get("_progress_callback")
        if request.method.family == "perturbation" and callable(progress_callback):
            progress_callback({"current": 0, "total": progress_total})

        hook_context = hook_manager if hook_manager is not None else _NullContext()
        with hook_context:
            for region in request.tile_plan.regions:
                if request.pause_waiter is not None:
                    request.pause_waiter(request.method, method_params)
                request.model.zero_grad(set_to_none=True)
                if img2.grad is not None:
                    img2.grad = None
                if hook_manager is not None:
                    hook_manager.clear()

                tile_input = img2[
                    ...,
                    region.slices[0],
                    region.slices[1],
                    region.slices[2],
                ]
                if request.method.family == "gradient":
                    tile_input.retain_grad()
                    logits = request.model(tile_input)
                else:
                    with torch.no_grad():
                        logits = request.model(tile_input)

                layers_by_name = (
                    hook_manager.layers_by_name() if hook_manager is not None else {}
                )
                tile_method_params = self._tile_method_params(
                    request=request,
                    base_params=method_params,
                    region=region,
                    patch_count=len(patches),
                    progress_total=progress_total,
                    progress_units=progress_units,
                    full_input=img2[0].detach(),
                )
                if request.pause_waiter is not None:
                    request.pause_waiter(request.method, tile_method_params)
                patches.append(
                    request.method.collect_patch_data(
                        CamPatchContext(
                            input_tensor=tile_input,
                            logits=logits,
                            layers_by_name=layers_by_name,
                            target_class=request.target_class,
                            objective=request.objective,
                            model=request.model,
                            method_params=tile_method_params,
                            device=request.device,
                        )
                    )
                )

        if logits is None:
            raise RuntimeError("沒有可用 tile 可供 XAI 計算。")
        layers = (
            _layer_channel_counts(patches[-1])
            if hook_manager is not None
            else {"input": 1}
        )
        return TileCollectionResult(
            patches=patches,
            layers=layers,
            model_output=logits.detach().to("cpu"),
            tile_plan=request.tile_plan,
        )

    @staticmethod
    def _tile_method_params(
        *,
        request: TileCollectionRequest,
        base_params: dict[str, object],
        region: TileRegion,
        patch_count: int,
        progress_total: int,
        progress_units: int,
        full_input: torch.Tensor,
    ) -> dict[str, object]:
        tile_method_params = dict(base_params)
        tile_method_params["_score_logger"] = request.logger
        tile_method_params["_tile_index"] = patch_count
        tile_method_params["_progress_total"] = progress_total
        tile_method_params["_progress_offset"] = patch_count * progress_units
        tile_method_params["_preview_spacing"] = request.model_input_spacing
        tile_method_params["_preview_metadata"] = {
            **dict(request.model_input_metadata or {}),
            "volume_id": "perturb-preview",
        }
        tile_method_params["_preview_full_input"] = full_input
        tile_method_params["_preview_tile_origin"] = region.origin
        tile_method_params["_preview_permute"] = request.display_permute
        if request.reference_mask is not None:
            tile_method_params["reference_mask"] = request.reference_mask[
                region.slices[0], region.slices[1], region.slices[2]
            ]
        return tile_method_params


def _layer_channel_counts(patch_payload: dict[str, object]) -> dict[str, int]:
    import torch

    layers = patch_payload.get("layers")
    if not isinstance(layers, dict):
        return {}
    counts: dict[str, int] = {}
    for name, layer_payload in layers.items():
        if not isinstance(layer_payload, dict):
            continue
        activation = layer_payload.get("activation")
        if isinstance(activation, torch.Tensor):
            counts[str(name)] = int(activation.size(1))
    return counts
