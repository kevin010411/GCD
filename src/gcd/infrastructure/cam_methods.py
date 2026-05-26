from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CamPatchContext:
    input_tensor: torch.Tensor
    logits: torch.Tensor
    layers_by_name: Mapping[str, torch.Tensor]
    target_class: int
    objective: Callable[[torch.Tensor, int], torch.Tensor]


class CamMethod(Protocol):
    id: str
    display_name: str
    category: str
    uses_layer_controls: bool

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]: ...

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        layer: str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int],
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor: ...


class LayerGradientCamMethod:
    id = ""
    display_name = ""
    category = "grad"
    uses_layer_controls = True

    def __init__(self, objective: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
        self._objective = objective

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
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

    def _layer_tensors(
        self,
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

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        layer: str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int],
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class GradCamMethod(LayerGradientCamMethod):
    id = "gradcam"
    display_name = "Grad-CAM"

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        layer: str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int],
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        activation, gradient = self._layer_tensors(patch_payload, layer)
        activation = activation[:, n1:n2, ...]
        gradient = gradient[:, n1:n2, ...]
        weights = torch.mean(gradient, dim=(2, 3, 4), keepdim=True)
        gradcam = torch.sum(activation * weights, dim=1, keepdim=True)
        return F.interpolate(gradcam, size=output_size, mode="trilinear")


class XResCamMethod(LayerGradientCamMethod):
    id = "xrescam"
    display_name = "XResCAM"

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        layer: str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int],
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        activation, gradient = self._layer_tensors(patch_payload, layer)
        xrescam = torch.sum(
            (activation * gradient)[:, n1:n2, ...], dim=1, keepdim=True
        )
        return F.interpolate(xrescam, size=output_size, mode="trilinear")


class GradCAMTestMethod:
    id = "gradcam_test"
    display_name = "Grad-CAM Test"
    category = "grad"
    uses_layer_controls = True

    @staticmethod
    def _objective(logits: torch.Tensor, target_class: int) -> torch.Tensor:
        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        return logits[0, target_class].sum()

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        loss = self._objective(context.logits, context.target_class)
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

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        layer: str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int],
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

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


class SaliencyMapMethod:
    """
    基於分類的SaliencyMap，拓展於三維分割
    _objective由外部傳入
    """

    id = "saliency_map"
    display_name = "Saliency Map"
    category = "grad"
    uses_layer_controls = False

    def __init__(self, objective: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
        self._objective = objective

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        loss = self._objective(context.logits, context.target_class)
        loss.backward()
        if context.input_tensor.grad is None:
            raise RuntimeError("input gradient 未產生，無法建立 Saliency Map payload。")
        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "input_gradient": context.input_tensor.grad.detach().to("cpu"),
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
        import torch
        import torch.nn.functional as F

        gradient = patch_payload["input_gradient"]
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("Saliency Map payload 缺少 input_gradient tensor。")

        saliency = torch.amax(torch.abs(gradient), dim=1, keepdim=True)
        return F.interpolate(saliency, size=output_size, mode="trilinear")


class PerturbationOcclusionMethod:
    id = "perturb_occlusion"
    display_name = "Occlusion"
    category = "perturbation"
    uses_layer_controls = True

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "layers": {
                key: {
                    "activation": value.detach().to("cpu"),
                }
                for key, value in context.layers_by_name.items()
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
        import torch
        import torch.nn.functional as F

        layers = patch_payload["layers"]
        if not isinstance(layers, dict) or layer not in layers:
            raise KeyError(f"layer '{layer}' 不存在於 CAM payload 中。")
        layer_payload = layers[layer]
        if not isinstance(layer_payload, dict):
            raise TypeError("CAM layer payload 格式錯誤。")
        activation = layer_payload["activation"]
        if not isinstance(activation, torch.Tensor):
            raise TypeError("CAM layer payload 缺少 activation tensor。")
        occlusion_map = torch.mean(
            torch.abs(activation[:, n1:n2, ...]), dim=1, keepdim=True
        )
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
