from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from .base import CamPatchContext, XaiLayerSelection, XaiMethod

if TYPE_CHECKING:
    import torch


class ScoreCamMethod(XaiMethod):
    """Gradient-free Score-CAM for 3D segmentation feature maps."""

    id = "scorecam"
    display_name = "Score-CAM"
    family = "gradient"
    uses_layer_controls = True
    uses_objective = True

    def __init__(self, objective: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
        self._objective = objective

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        import torch
        import torch.nn.functional as F

        if context.model is None:
            raise RuntimeError("Score-CAM 需要可執行的模型。")

        params = dict(context.method_params or {})
        requested_layer = str(params.get("_selected_layer", "") or "")
        layer = (
            requested_layer
            if requested_layer in context.layers_by_name
            else next(iter(context.layers_by_name), "")
        )
        if not layer:
            raise RuntimeError("目前沒有可用的 Score-CAM layer。")

        activation = context.layers_by_name[layer].detach()
        channel_count = int(activation.size(1))
        start = max(0, min(int(params.get("_feature_start", 0)), channel_count))
        stop = max(start, min(int(params.get("_feature_stop", channel_count)), channel_count))
        if stop <= start:
            raise ValueError("Score-CAM feature 範圍不可為空。")

        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for channel in range(start, stop):
                mask = F.interpolate(
                    activation[:, channel : channel + 1],
                    size=context.input_tensor.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
                flat = mask.flatten(start_dim=2)
                minimum = flat.amin(dim=2, keepdim=True).view(mask.size(0), 1, 1, 1, 1)
                maximum = flat.amax(dim=2, keepdim=True).view(mask.size(0), 1, 1, 1, 1)
                mask = (mask - minimum) / torch.clamp_min(maximum - minimum, 1e-12)
                masked_logits = context.model(context.input_tensor.detach() * mask)
                score = context.objective(masked_logits, context.target_class)
                scores.append(score.detach().reshape(()).to("cpu"))

        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "selected_layer": layer,
            "feature_start": start,
            "feature_stop": stop,
            "scores": torch.stack(scores),
            "layers": {
                name: {"activation": value.detach().to("cpu")}
                for name, value in context.layers_by_name.items()
            },
        }

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        import torch
        import torch.nn.functional as F

        selected_layer = str(patch_payload.get("selected_layer", ""))
        if selection.layer != selected_layer:
            raise ValueError(
                f"Score-CAM payload 是針對 layer '{selected_layer}' 計算，"
                f"無法改用 '{selection.layer}'。"
            )
        layers = patch_payload.get("layers")
        if not isinstance(layers, dict) or selection.layer not in layers:
            raise KeyError(f"layer '{selection.layer}' 不存在於 Score-CAM payload 中。")
        layer_payload = layers[selection.layer]
        if not isinstance(layer_payload, dict):
            raise TypeError("Score-CAM layer payload 格式錯誤。")
        activation = layer_payload.get("activation")
        scores = patch_payload.get("scores")
        if not isinstance(activation, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("Score-CAM payload 缺少 activation/scores tensor。")

        stored_start = int(patch_payload.get("feature_start", 0))
        stored_stop = int(patch_payload.get("feature_stop", activation.size(1)))
        start = max(stored_start, int(selection.n1))
        stop = min(stored_stop, int(selection.n2))
        if stop <= start:
            raise ValueError("指定的 Score-CAM feature 範圍未預先計算。")
        score_slice = scores[start - stored_start : stop - stored_start]
        weights = torch.softmax(score_slice.to(dtype=activation.dtype), dim=0).view(
            1, -1, 1, 1, 1
        )
        cam = torch.sum(activation[:, start:stop] * weights, dim=1, keepdim=True)
        return F.interpolate(
            cam,
            size=selection.output_size,
            mode="trilinear",
            align_corners=False,
        )
