from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from .base import CamPatchContext, XaiLayerSelection, XaiMethod

if TYPE_CHECKING:
    import torch


class SaliencyMapMethod(XaiMethod):
    """
    基於分類的 Saliency Map，拓展於三維分割。
    """

    id = "saliency_map"
    display_name = "Saliency Map"
    family = "gradient"
    uses_layer_controls = False
    uses_objective = True

    def __init__(self, objective: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
        self._objective = objective

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        loss = context.objective(context.logits, context.target_class)
        loss.backward()
        if context.input_tensor.grad is None:
            raise RuntimeError("input gradient 未產生，無法建立 Saliency Map payload。")
        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "input_gradient": context.input_tensor.grad.detach().to("cpu"),
        }

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        import torch
        import torch.nn.functional as F

        gradient = patch_payload["input_gradient"]
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("Saliency Map payload 缺少 input_gradient tensor。")
        saliency = torch.amax(torch.abs(gradient), dim=1, keepdim=True)
        return F.interpolate(saliency, size=selection.output_size, mode="trilinear")
