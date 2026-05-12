from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import torch
import torch.nn.functional as F


class CamMethod(Protocol):
    id: str
    display_name: str
    category: str

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
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        ...


class GradCamMethod:
    id = "gradcam"
    display_name = "Grad-CAM"
    category = "grad"

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
        method_params: Mapping[str, object] | None = None,
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


class GradCAMTestMethod:
    id = "gradcam_test"
    display_name = "Grad-CAM Test"
    category = "grad"

    @staticmethod
    def _objective(logits: torch.Tensor, target_class: int) -> torch.Tensor:
        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        return logits[0, target_class].sum()

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
        method_params: Mapping[str, object] | None = None,
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

        test_cam = torch.sum(
            (activation * gradient)[:, n1:n2, ...], dim=1, keepdim=True
        )
        return F.interpolate(test_cam, size=output_size, mode="trilinear")


class PerturbationOcclusionMethod:
    id = "perturb_occlusion"
    display_name = "Occlusion"
    category = "perturbation"

    def collect_patch_data(
        self,
        layers_by_name: Mapping[str, torch.Tensor],
        logits: torch.Tensor,
        target_class: int,
    ) -> dict[str, object]:
        del target_class
        return {
            "method": self.id,
            "pred": logits.detach().to("cpu"),
            "layers": {
                key: {
                    "activation": value.detach().to("cpu"),
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
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        layers = patch_payload["layers"]
        if not isinstance(layers, dict) or layer not in layers:
            raise KeyError(f"layer '{layer}' 不存在於 CAM payload 中。")
        layer_payload = layers[layer]
        if not isinstance(layer_payload, dict):
            raise TypeError("CAM layer payload 格式錯誤。")
        activation = layer_payload["activation"]
        if not isinstance(activation, torch.Tensor):
            raise TypeError("CAM layer payload 缺少 activation tensor。")
        occlusion_map = torch.mean(torch.abs(activation[:, n1:n2, ...]), dim=1, keepdim=True)
        occlusion_map = F.interpolate(occlusion_map, size=output_size, mode="trilinear")
        block_size = int((method_params or {}).get("block_size", 16) or 16)
        kernel = max(1, min(block_size // 8, 7))
        if kernel > 1:
            occlusion_map = F.avg_pool3d(
                occlusion_map,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
        return occlusion_map
