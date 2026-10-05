from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..methods.cam_methods import CamMethod, XaiLayerSelection
from ..tiling.tile_strategy import SlidingWindowTileStrategy, TilePlan
from ..tiling.scorecam_blend import tile_importance_map

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
        tile_plan = request.tile_plan or SlidingWindowTileStrategy().plan(
            input_shape=request.img1[0].shape,
            patch_size=request.size,
            stride=request.stride,
        )

        cam = torch.zeros(shape, dtype=torch.float32)
        coverage = torch.zeros(shape, dtype=torch.float32)
        if not isinstance(patches[0].get("pred"), torch.Tensor):
            raise TypeError("CAM patch payload 缺少 pred tensor。")
        pred_shape = list(patches[0]["pred"].shape)[:2] + shape
        model_out = torch.zeros(pred_shape, dtype=torch.float32)
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

            weight = tile_importance_map(
                region.size, request.method_params or {}, default_mode="constant",
            ).unsqueeze(0).unsqueeze(0)
            q *= weight
            p1 *= weight

            xs, ys, zs = region.slices

            cam[xs, ys, zs] += q[0, 0]
            model_out[:, :, xs, ys, zs] += p1
            if coverage is not None:
                coverage[xs, ys, zs] += weight[0, 0]

        if coverage is not None:
            coverage.clamp_min_(1e-12)
            cam /= coverage
            model_out /= coverage.unsqueeze(0).unsqueeze(0)
            del coverage

        perturb_signal_max = None
        if method.family == "perturbation":
            cam.abs_()
            perturb_signal_max = torch.max(cam)
        else:
            cam.clamp_min_(0)
        cam -= torch.min(cam)
        maximum = torch.max(cam)
        minimum_signal = (torch.finfo(torch.float32).eps
                          if method.id in {"gradcam", "hirescam", "xrescam", "layercam", "scorecam"} else 0)
        if maximum > minimum_signal:
            cam /= maximum
        elif (
            method.family == "perturbation"
            and perturb_signal_max is not None
            and perturb_signal_max > 0
        ):
            cam = torch.ones_like(cam)
        else:
            cam.zero_()

        prediction_dtype = self._prediction_dtype(model_out.shape[1])
        model_output = torch.argmax(model_out, dim=1)[0].to(prediction_dtype)
        del model_out

        return XaiCamRunResult(
            selected_layer=selected_layer,
            layers=layers,
            cam=cam.permute(*request.permute),
            model_output=model_output.permute(*request.permute),
        )

    @staticmethod
    def _prediction_dtype(class_count: int):
        """Return the smallest signed/unsigned dtype that can hold class IDs."""
        import torch

        if class_count <= 256:
            return torch.uint8
        if class_count <= 32_768:
            return torch.int16
        return torch.int32
