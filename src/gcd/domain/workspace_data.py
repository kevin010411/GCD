from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .transfer_function import DataRange, TransferFunction


@dataclass(frozen=True)
class DatasetInput:
    img0: object
    img1: object
    origin_img: object
    origin_meta: dict[str, object]
    origin_shape: tuple[int, ...] | None
    img1_spacing: tuple[float, float, float]
    display_metadata: dict[str, object]
    layers: dict[str, int]
    file_name: str
    target_class: int
    active_method_id: str
    active_objective_id: str = "predicted_target_mask"
    xai_cache_key: str = ""
    raw_display_data: object = None
    raw_spacing: tuple[float, float, float] | None = None
    raw_display_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    name: str
    file_name: str
    input_state: DatasetInput
    layer_names: tuple[str, ...]
    selected_layer: str
    feature_size: int
    base_volume_id: str
    result_ids: tuple[str, ...]
    base_shape: tuple[int, ...]
    base_spacing: tuple[float, float, float]
    display_metadata: dict[str, object]

    def __getitem__(self, key: str) -> object:
        aliases = {
            "base_item_id": self.base_volume_id,
            "result_ids": list(self.result_ids),
            "layer_names": list(self.layer_names),
            "selected_layer": self.selected_layer,
            "feature_size": self.feature_size,
            "name": self.name,
            "file_name": self.file_name,
            "base_shape": self.base_shape,
            "base_spacing": self.base_spacing,
            "display_metadata": self.display_metadata,
        }
        return aliases[key]

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def with_result_id(self, result_id: str) -> DatasetRecord:
        if result_id in self.result_ids:
            return self
        return replace(self, result_ids=(*self.result_ids, result_id))


@dataclass(frozen=True)
class VolumeRecord:
    id: str
    dataset_id: str
    display_name: str
    source: str
    method_id: str
    data: object
    data_range: DataRange
    transfer_function: TransferFunction
    spacing: tuple[float, float, float]
    metadata: dict[str, object]
    shape: tuple[int, ...]
    source_base_item_id: str
    source_shape: tuple[int, ...]
    source_spacing: tuple[float, float, float]
    source_affine: object
    visible: bool = True
    plugin_metadata: dict[str, object] | None = None

    def to_render_item(self, *, is_focus: bool = False) -> dict[str, object]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "source": self.source,
            "method_id": self.method_id,
            "data": self.data,
            "data_range": self.data_range,
            "transfer_function": self.transfer_function,
            "visible": self.visible,
            "spacing": self.spacing,
            "metadata": self.metadata,
            "shape": self.shape,
            "source_base_item_id": self.source_base_item_id,
            "source_shape": self.source_shape,
            "source_spacing": self.source_spacing,
            "source_affine": self.source_affine,
            "is_focus": is_focus,
        }


@dataclass(frozen=True)
class SelectionState:
    active_dataset_by_plugin: dict[str, str]
    selected_volume_id: str = ""
    volume_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class XaiComputeRequest:
    target_class: int
    layer: str | None
    n1: int
    n2: int
    method: str
    result_name: str
    objective_id: str = "predicted_target_mask"
    method_params: dict[str, object] | None = None


@dataclass(frozen=True)
class XaiComputeResult:
    dataset_input: DatasetInput
    layer_names: tuple[str, ...]
    selected_layer: str
    method_options: tuple[dict[str, object], ...]
    selected_method: str
    objective_options: tuple[dict[str, object], ...]
    selected_objective: str
    feature_size: int
    volume: VolumeRecord
    volume_data_range: DataRange


@dataclass(frozen=True)
class WorkspaceEvent:
    name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class DatasetAdded(WorkspaceEvent):
    def __init__(self, dataset_id: str) -> None:
        super().__init__("dataset_added", {"dataset_id": dataset_id})


@dataclass(frozen=True)
class VolumeUpserted(WorkspaceEvent):
    def __init__(self, volume_id: str, dataset_id: str) -> None:
        super().__init__(
            "volume_upserted", {"volume_id": volume_id, "dataset_id": dataset_id}
        )


@dataclass(frozen=True)
class VolumeVisibilityChanged(WorkspaceEvent):
    def __init__(self, volume_id: str, visible: bool) -> None:
        super().__init__(
            "volume_visibility_changed",
            {"volume_id": volume_id, "visible": bool(visible)},
        )


@dataclass(frozen=True)
class VolumeRenamed(WorkspaceEvent):
    def __init__(self, volume_id: str) -> None:
        super().__init__("volume_renamed", {"volume_id": volume_id})


@dataclass(frozen=True)
class VolumeDeleted(WorkspaceEvent):
    def __init__(self, volume_id: str, dataset_id: str) -> None:
        super().__init__(
            "volume_deleted", {"volume_id": volume_id, "dataset_id": dataset_id}
        )


@dataclass(frozen=True)
class DatasetDeleted(WorkspaceEvent):
    def __init__(self, dataset_id: str) -> None:
        super().__init__("dataset_deleted", {"dataset_id": dataset_id})


@dataclass(frozen=True)
class VolumeOrderChanged(WorkspaceEvent):
    def __init__(self) -> None:
        super().__init__("volume_order_changed", {})


@dataclass(frozen=True)
class TransferChanged(WorkspaceEvent):
    def __init__(self, volume_id: str) -> None:
        super().__init__("transfer_changed", {"volume_id": volume_id})


@dataclass(frozen=True)
class SelectionChanged(WorkspaceEvent):
    def __init__(self, selected_volume_id: str) -> None:
        super().__init__("selection_changed", {"selected_volume_id": selected_volume_id})
