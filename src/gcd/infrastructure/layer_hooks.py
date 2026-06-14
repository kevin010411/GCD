from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


class XaiLayerHookManager:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles = []
        self._layers: dict[str, torch.Tensor] = {}
        self._targets = self._validate_targets(model)

    @staticmethod
    def _validate_targets(model: nn.Module) -> dict[str, str]:
        targets = getattr(model, "xai_layer_targets", None)
        if not isinstance(targets, Mapping) or not targets:
            raise RuntimeError(
                "模型缺少 xai_layer_targets。請在模型 class/instance 宣告 "
                "dict[str, str]，例如 {'dec4': 'decoder4'}。"
            )
        normalized = {str(name): str(path) for name, path in targets.items()}
        for name, path in normalized.items():
            try:
                model.get_submodule(path)
            except AttributeError as exc:
                raise RuntimeError(
                    f"模型 xai_layer_targets 中的 layer '{name}' 指向不存在的 "
                    f"module path '{path}'。"
                ) from exc
        return normalized

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self._targets.keys())

    def __enter__(self) -> XaiLayerHookManager:
        for name, path in self._targets.items():
            module = self.model.get_submodule(path)
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.clear()

    def clear(self) -> None:
        self._layers = {}

    def layers_by_name(self) -> dict[str, torch.Tensor]:
        missing = [name for name in self._targets if name not in self._layers]
        if missing:
            raise RuntimeError(
                "XAI hook 未擷取到 layer activation: " + ", ".join(missing)
            )
        return dict(self._layers)

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output) -> None:
            import torch

            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"XAI layer '{name}' 的 output 不是 torch.Tensor；"
                    "第一版 hook 只支援單一 tensor output。"
                )
            if output.requires_grad:
                output.retain_grad()
            self._layers[name] = output

        return hook
