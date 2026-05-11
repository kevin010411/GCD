from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import torch
import torch.nn.functional as F


class CamMethod(Protocol):
    id: str
    display_name: str

    def collect_patch_data(
        self,
        layers_by_name: Mapping[str, torch.Tensor],
        logits: torch.Tensor,
        target_class: int,
    ) -> dict[str, object]:
        ...

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        layer: str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int],
    ) -> torch.Tensor:
        ...


class GradCamMethod:
    id = "gradcam"
    display_name = "Grad-CAM"

    def __init__(
        self, objective: Callable[[torch.Tensor, int], torch.Tensor]
    ) -> None:
        self._objective = objective

    def collect_patch_data(
        self,
        layers_by_name: Mapping[str, torch.Tensor],
        logits: torch.Tensor,
        target_class: int,
    ) -> dict[str, object]:
        loss = self._objective(logits, target_class)
        loss.backward()
        return {
            "method": self.id,
            "pred": logits.detach().to("cpu"),
            "layers": {
                key: {
                    "activation": value.detach().to("cpu"),
                    "gradient": value.grad.detach().to("cpu"),
                }
                for key, value in layers_by_name.items()
            },
        }

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        layer: str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int],
    ) -> torch.Tensor:
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

        gradcam = torch.sum(
            (activation * gradient)[:, n1:n2, ...], dim=1, keepdim=True
        )
        return F.interpolate(gradcam, size=output_size, mode="trilinear")
