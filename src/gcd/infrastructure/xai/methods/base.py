from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CamPatchContext:
    input_tensor: torch.Tensor
    logits: torch.Tensor
    layers_by_name: Mapping[str, torch.Tensor]
    target_class: int
    objective: Callable[[torch.Tensor, int], torch.Tensor]
    model: Any = None
    method_params: Mapping[str, object] | None = None
    device: Any = None


@dataclass(frozen=True)
class XaiLayerSelection:
    layer: str
    n1: int
    n2: int
    output_size: tuple[int, int, int]


@dataclass(frozen=True)
class XaiParameterSpec:
    id: str
    label: str
    kind: str
    default: object = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    choices: tuple[tuple[str, object], ...] = ()
    tooltip: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "tooltip": self.tooltip,
        }
        if self.min_value is not None:
            payload["min"] = self.min_value
        if self.max_value is not None:
            payload["max"] = self.max_value
        if self.step is not None:
            payload["step"] = self.step
        if self.choices:
            payload["choices"] = [
                {"label": label, "value": value} for label, value in self.choices
            ]
        return payload


@dataclass(frozen=True)
class XaiMethodDefinition:
    id: str
    name: str
    family: str
    uses_layer_controls: bool
    uses_objective: bool
    parameters: tuple[XaiParameterSpec, ...] = ()

    def to_option(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "uses_layer_controls": self.uses_layer_controls,
            "uses_objective": self.uses_objective,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


class XaiMethod(ABC):
    id = ""
    display_name = ""
    family = ""
    uses_layer_controls = True
    uses_objective = True

    def parameter_schema(self) -> Sequence[XaiParameterSpec]:
        return ()

    def definition(self) -> XaiMethodDefinition:
        return XaiMethodDefinition(
            id=self.id,
            name=self.display_name,
            family=self.family,
            uses_layer_controls=bool(self.uses_layer_controls),
            uses_objective=bool(self.uses_objective),
            parameters=tuple(self.parameter_schema()),
        )

    @abstractmethod
    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        raise NotImplementedError

    def build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection_or_layer: XaiLayerSelection | str,
        n1: int = 0,
        n2: int = 999,
        output_size: tuple[int, int, int] | None = None,
        method_params: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        selection = self._coerce_selection(selection_or_layer, n1, n2, output_size)
        return self._build_tile_cam(patch_payload, selection, method_params or {})

    @staticmethod
    def _coerce_selection(
        selection_or_layer: XaiLayerSelection | str,
        n1: int,
        n2: int,
        output_size: tuple[int, int, int] | None,
    ) -> XaiLayerSelection:
        if isinstance(selection_or_layer, XaiLayerSelection):
            return selection_or_layer
        if output_size is None:
            raise ValueError("output_size is required when layer is passed directly.")
        return XaiLayerSelection(str(selection_or_layer), int(n1), int(n2), output_size)

    @abstractmethod
    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ) -> torch.Tensor:
        raise NotImplementedError
