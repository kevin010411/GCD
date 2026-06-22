from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..methods.cam_methods import CamMethod, XaiLayerSelection
from ..tiling.tile_strategy import LegacyFourTileStrategy, TilePlan

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class XaiCamRunRequest:
    method: CamMethod
    patches: Sequence[dict[str, object]]
    layers: Mapping[str, int]
    img1: torch.Tensor
    size: int
    stride: int
    permute: tuple[int, int, int]
    default_layer: str
    tile_plan: TilePlan | None = None
    layer: str | None = None
    n1: int = 0
    n2: int = 999
    method_params: Mapping[str, object] | None = None
    logger: Callable[[str], None] | None = None


@dataclass(frozen=True)
class XaiCamRunResult:
    selected_layer: str
    layers: dict[str, int]
    cam: torch.Tensor
    model_output: torch.Tensor


class XaiCamRunner:
    """Build a full-volume CAM from prepared XAI patch payloads."""

    def run(self, request: XaiCamRunRequest) -> XaiCamRunResult:
        import torch
        import torch.nn.functional as F

        method = request.method
        patches = list(request.patches)
        if not patches:
            raise RuntimeError("尚未準備 XAI patch 資料，請先執行 prepare_xai_inputs。")
        if any(
            not isinstance(item, dict) or item.get("method") != method.id
            for item in patches
        ):
            raise ValueError("目前的 CAM patch payload 與指定 method 不一致。")

        layers = dict(request.layers)
        if method.uses_layer_controls:
            available_layers = list(layers.keys())
            selected_layer = request.layer or request.default_layer
            if selected_layer not in layers:
                fallback_layer = (
                    request.default_layer
                    if request.default_layer in layers
                    else (available_layers[0] if available_layers else "")
                )
                if not fallback_layer:
                    raise RuntimeError("目前沒有可用的 CAM layer。")
                selected_layer = fallback_layer
                if request.logger is not None:
                    request.logger(f"指定 layer 不存在，改用可用 layer: {selected_layer}")
        else:
            selected_layer = "input"
            layers = {"input": 1}

        n2 = min(int(request.n2), int(layers[selected_layer]))
        shape = list(request.img1[0].shape)
        cam = torch.zeros(shape, dtype=torch.float32)
        coverage = torch.zeros(shape, dtype=torch.float32)
        if not isinstance(patches[0].get("pred"), torch.Tensor):
            raise TypeError("CAM patch payload 缺少 pred tensor。")
        pred_shape = list(patches[0]["pred"].shape)[:2] + shape
        model_out = torch.zeros(pred_shape, dtype=torch.float32)
        tile_plan = request.tile_plan or LegacyFourTileStrategy().plan(
            input_shape=request.img1[0].shape,
            patch_size=request.size,
            stride=request.stride,
        )
        legacy_blending = tile_plan.strategy_id == LegacyFourTileStrategy.id
        for index, region in enumerate(tile_plan.regions):
            q = method.build_tile_cam(
                patches[index],
                XaiLayerSelection(
                    selected_layer,
                    int(request.n1),
                    n2,
                    region.size,
                ),
                method_params=request.method_params,
            )

            p1 = patches[index]["pred"]
            if not isinstance(p1, torch.Tensor):
                raise TypeError("CAM patch payload 缺少 pred tensor。")
            p1 = F.interpolate(
                p1, size=region.size, mode="trilinear"
            )

            if legacy_blending:
                self._apply_tile_blend(q, p1, index, request.size, request.stride)

            xs, ys, zs = region.slices

            cam[xs, ys, zs] += q[0, 0]
            model_out[:, :, xs, ys, zs] += p1
            if not legacy_blending:
                coverage[xs, ys, zs] += 1

        if not legacy_blending:
            coverage = torch.clamp_min(coverage, 1)
            cam /= coverage
            model_out /= coverage.unsqueeze(0).unsqueeze(0)

        perturb_signal_max = None
        if method.family == "perturbation":
            cam = torch.abs(cam)
            perturb_signal_max = torch.max(cam)
        else:
            cam = torch.maximum(cam, torch.tensor(0))
        cam -= torch.min(cam)
        maximum = torch.max(cam)
        if maximum > 0:
            cam /= maximum
        elif (
            method.family == "perturbation"
            and perturb_signal_max is not None
            and perturb_signal_max > 0
        ):
            cam = torch.ones_like(cam)

        return XaiCamRunResult(
            selected_layer=selected_layer,
            layers=layers,
            cam=cam.permute(*request.permute),
            model_output=torch.argmax(model_out, dim=1)[0].permute(*request.permute),
        )

    @staticmethod
    def _apply_tile_blend(
        q: torch.Tensor, p1: torch.Tensor, index: int, size: int, stride: int
    ) -> None:
        overlap = size - stride
        if index in (0, 1):
            for i in range(overlap):
                weight = (overlap - i) / overlap
                q[0, 0, stride + i, :, :] *= weight
                p1[0, 0, stride + i, :, :] *= weight
        if index in (2, 3):
            for i in range(overlap):
                weight = i / overlap
                q[0, 0, i, :, :] *= weight
                p1[0, 0, i, :, :] *= weight
        if index in (0, 2):
            for i in range(overlap):
                weight = (overlap - i) / overlap
                q[0, 0, :, stride + i, :] *= weight
                p1[0, 0, :, stride + i, :] *= weight
        if index in (1, 3):
            for i in range(overlap):
                weight = i / overlap
                q[0, 0, :, i, :] *= weight
                p1[0, 0, :, i, :] *= weight
