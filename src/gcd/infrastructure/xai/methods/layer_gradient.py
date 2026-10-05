from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import CamPatchContext, XaiMethod

if TYPE_CHECKING:
    import torch


class LayerGradientXaiMethod(XaiMethod):
    family = "gradient"
    uses_layer_controls = True
    uses_objective = True

    def __init__(self, objective: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
        self._objective = objective

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        import torch
        params = context.method_params or {}
        from ..cam_protocol import resolve_cam_protocol, target_score
        protocol = resolve_cam_protocol(params.get("cam_protocol"))
        if params.get("_objective_id") == "predicted_target_mask":
            target = params.get("_fixed_target_mask")
            target = (context.logits.detach().argmax(dim=1) == context.target_class
                      if target is None else torch.as_tensor(target, device=context.logits.device, dtype=torch.bool))
            # A prediction made under inference_mode cannot be saved by autograd.
            target = target.clone()
            values = context.logits[:, context.target_class]
            if target.shape != values.shape:
                raise ValueError("CAM 固定目標 mask 與 tile logits shape 不一致。")
            # Keep a differentiable zero for empty targets: never average an empty tensor.
            loss = target_score(context.logits, context.target_class, target, protocol["reduction"])
        else:
            loss = context.objective(context.logits, context.target_class)
        loss.backward()
        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "layers": {
                key: {
                    "activation": value.detach().to("cpu"),
                    "gradient": self._required_gradient(value),
                }
                for key, value in context.layers_by_name.items()
            },
        }

    @staticmethod
    def _required_gradient(value: torch.Tensor) -> torch.Tensor:
        if value.grad is None:
            raise RuntimeError("CAM layer gradient 未產生，無法建立 Grad-CAM payload。")
        return value.grad.detach().to("cpu")

    @staticmethod
    def _layer_tensors(
        patch_payload: dict[str, object],
        layer: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        layers = patch_payload["layers"]
        if not isinstance(layers, dict) or layer not in layers:
            raise KeyError(f"layer '{layer}' 不存在於 CAM payload 中。")
        layer_payload = layers[layer]
        if not isinstance(layer_payload, dict):
            raise TypeError("CAM layer payload 格式錯誤。")
        activation = layer_payload["activation"]
        gradient = layer_payload["gradient"]
        if not isinstance(activation, torch.Tensor) or not isinstance(
            gradient, torch.Tensor
        ):
            raise TypeError("CAM layer payload 缺少 activation/gradient tensor。")
        return activation, gradient
