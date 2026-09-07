from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..domain import DataRange, TransferFunction
from ..domain.workspace_data import (
    DatasetAdded,
    DatasetDeleted,
    DatasetInput,
    DatasetRecord,
    SelectionChanged,
    SelectionState,
    TransferChanged,
    VolumeOrderChanged,
    VolumeRenamed,
    VolumeRecord,
    VolumeDeleted,
    VolumeUpserted,
    VolumeVisibilityChanged,
    WorkspaceEvent,
    XaiComputeResult,
)


EventHandler = Callable[[WorkspaceEvent], None]


class WorkspaceDataStore:
    def __init__(self) -> None:
        self.datasets: dict[str, DatasetRecord] = {}
        self.volumes: dict[str, VolumeRecord] = {}
        self.selection = SelectionState(active_dataset_by_plugin={})
        self._subscribers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    def _emit(self, event: WorkspaceEvent) -> None:
        for handler in list(self._subscribers):
            handler(event)

    @property
    def dataset_order(self) -> list[str]:
        return list(self.datasets)

    @property
    def render_items(self) -> dict[str, dict[str, object]]:
        return {
            volume_id: volume.to_render_item(
                is_focus=volume_id == self.selection.selected_volume_id
            )
            for volume_id, volume in self.volumes.items()
        }

    @property
    def volume_order(self) -> list[str]:
        return list(self.selection.volume_order)

    @property
    def volume_visibility(self) -> dict[str, bool]:
        return {
            volume_id: self.volumes[volume_id].visible
            for volume_id in self.selection.volume_order
            if volume_id in self.volumes
        }

    @property
    def selected_transfer_volume_id(self) -> str:
        return self.selection.selected_volume_id

    @selected_transfer_volume_id.setter
    def selected_transfer_volume_id(self, value: str) -> None:
        self.set_selected_transfer_volume(value)

    def add_loaded_dataset(self, dataset_id: str, result: dict[str, Any]) -> str:
        dataset_name = Path(str(result["file_name"])).stem
        base_item_id = f"{dataset_id}:base"
        input_state = result["dataset_input"]
        if not isinstance(input_state, DatasetInput):
            raise TypeError("load_input result must include DatasetInput as dataset_input")
        base_shape = tuple(int(v) for v in result["volume_data"].shape)
        base_spacing = tuple(float(v) for v in result["spacing"])
        display_metadata = dict(result["display_metadata"])
        source_shape = tuple(
            int(v) for v in display_metadata.get("source_shape", base_shape)
        )
        source_affine = display_metadata.get(
            "source_affine", display_metadata.get("affine")
        )
        self.datasets[dataset_id] = DatasetRecord(
            id=dataset_id,
            name=dataset_name,
            file_name=str(result["file_name"]),
            input_state=input_state,
            layer_names=tuple(str(v) for v in result["layer_names"]),
            selected_layer=str(result["selected_layer"]),
            feature_size=int(result["feature_size"]),
            base_volume_id=base_item_id,
            result_ids=(),
            base_shape=base_shape,
            base_spacing=base_spacing,
            display_metadata=display_metadata,
        )
        self.volumes[base_item_id] = VolumeRecord(
            id=base_item_id,
            dataset_id=dataset_id,
            display_name=dataset_name,
            source="base",
            method_id="base",
            data=result["volume_data"],
            data_range=result["volume_data_range"],
            transfer_function=result["volume_transfer_function"],
            spacing=base_spacing,
            metadata={**display_metadata, "volume_id": base_item_id},
            shape=base_shape,
            source_base_item_id=base_item_id,
            source_shape=source_shape,
            source_spacing=base_spacing,
            source_affine=source_affine,
        )
        volume_order = (*self.selection.volume_order, base_item_id)
        active = dict(self.selection.active_dataset_by_plugin)
        active["gradcam"] = dataset_id
        active["perturbation"] = dataset_id
        selected = self.selection.selected_volume_id or base_item_id
        self.selection = SelectionState(active, selected, volume_order)
        self._emit(DatasetAdded(dataset_id))
        self._emit(VolumeUpserted(base_item_id, dataset_id))
        return dataset_id

    def active_dataset_id(self, plugin_id: str) -> str:
        return self.selection.active_dataset_by_plugin.get(plugin_id, "")

    def set_active_dataset(self, plugin_id: str, dataset_id: str) -> None:
        active = dict(self.selection.active_dataset_by_plugin)
        active[plugin_id] = dataset_id
        self.selection = replace(self.selection, active_dataset_by_plugin=active)
        self._emit(SelectionChanged(self.selection.selected_volume_id))

    def dataset_options(self) -> list[dict[str, str]]:
        return [
            {"id": dataset_id, "name": str(self.datasets[dataset_id].name)}
            for dataset_id in self.dataset_order
        ]

    def volume_options(self) -> list[dict[str, str]]:
        return [
            {"id": volume_id, "name": str(self.volume_display_name(volume_id))}
            for volume_id in self.volume_order
            if volume_id in self.volumes
        ]

    def current_transfer_target(self) -> str:
        return self.selection.selected_volume_id

    def current_transfer_state(self) -> tuple[TransferFunction, DataRange]:
        current = self.volumes.get(self.current_transfer_target())
        if current is None:
            return TransferFunction.base_preset(), DataRange(0.0, 1.0)
        return current.transfer_function, current.data_range

    def update_current_transfer_state(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> bool:
        volume_id = self.current_transfer_target()
        current = self.volumes.get(volume_id)
        if current is None:
            return False
        self.volumes[volume_id] = replace(
            current, transfer_function=transfer_function, data_range=data_range
        )
        self._emit(TransferChanged(volume_id))
        return True

    def volume_visible(self, volume_id: str) -> bool:
        volume = self.volumes.get(volume_id)
        return True if volume is None else bool(volume.visible)

    def set_volume_visibility(self, volume_id: str, visible: bool) -> None:
        volume = self.volumes.get(volume_id)
        if volume is None:
            return
        self.volumes[volume_id] = replace(volume, visible=bool(visible))
        self._emit(VolumeVisibilityChanged(volume_id, bool(visible)))

    def set_volume_order(self, ordered_ids: list[str]) -> None:
        if not ordered_ids:
            return
        self.selection = replace(self.selection, volume_order=tuple(ordered_ids))
        self._emit(VolumeOrderChanged())

    def volume_display_name(self, volume_id: str) -> str:
        volume = self.volumes.get(volume_id)
        return volume_id if volume is None else str(volume.display_name)

    def rename_item(self, volume_id: str, name: str) -> bool:
        volume = self.volumes.get(volume_id)
        if volume is None:
            return False
        self.volumes[volume_id] = replace(volume, display_name=name)
        self._emit(VolumeRenamed(volume_id))
        return True

    def set_selected_transfer_volume(self, volume_id: str) -> None:
        self.selection = replace(self.selection, selected_volume_id=volume_id)
        self._emit(SelectionChanged(volume_id))

    def ordered_volume_payloads(self) -> list[object]:
        return [
            self.volumes[volume_id].data
            for volume_id in self.selection.volume_order
            if volume_id in self.volumes
        ]

    def ordered_volume_spacing(self) -> list[tuple[float, float, float]]:
        return [
            self.volumes[volume_id].spacing
            for volume_id in self.selection.volume_order
            if volume_id in self.volumes
        ]

    def ordered_volume_metadata(self) -> list[dict[str, object]]:
        return [
            dict(self.volumes[volume_id].metadata)
            for volume_id in self.selection.volume_order
            if volume_id in self.volumes
        ]

    def volume_list_items(self) -> list[dict[str, object]]:
        return [
            {
                "id": volume_id,
                "display_name": self.volume_display_name(volume_id),
                "visible": self.volume_visible(volume_id),
            }
            for volume_id in self.selection.volume_order
            if volume_id in self.volumes
        ]

    def upsert_xai_result(
        self, dataset_id: str, result: dict[str, Any] | XaiComputeResult
    ) -> str | None:
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return None

        if isinstance(result, XaiComputeResult):
            compute_result = result
            source_volume = compute_result.volume
            volume_id = self.create_prediction_volume_id(
                dataset_id, source_volume.method_id
            )
            volume = replace(
                source_volume,
                id=volume_id,
                dataset_id=dataset_id,
                display_name=self.unique_volume_display_name(
                    source_volume.display_name
                ),
                metadata={**source_volume.metadata, "volume_id": volume_id},
                source_base_item_id=dataset.base_volume_id,
                source_shape=tuple(
                    int(v)
                    for v in source_volume.metadata.get(
                        "source_shape", dataset.display_metadata.get("source_shape", dataset.base_shape)
                    )
                ),
                source_spacing=dataset.base_spacing,
                source_affine=source_volume.metadata.get(
                    "source_affine",
                    dataset.display_metadata.get(
                        "source_affine", dataset.display_metadata.get("affine")
                    ),
                ),
            )
        else:
            compute_result = None
            item_payload = result["renderable_item"]
            method_id = str(item_payload["method_id"])
            volume_id = self.create_prediction_volume_id(dataset_id, method_id)
            volume = VolumeRecord(
                id=volume_id,
                dataset_id=dataset_id,
                display_name=self.unique_volume_display_name(item_payload["name"]),
                source=item_payload["source"],
                method_id=method_id,
                data=item_payload["data"],
                data_range=item_payload["data_range"],
                transfer_function=item_payload["transfer_function"],
                spacing=tuple(float(v) for v in item_payload["spacing"]),
                metadata={**item_payload["metadata"], "volume_id": volume_id},
                shape=tuple(int(v) for v in item_payload["shape"]),
                source_base_item_id=dataset.base_volume_id,
                source_shape=tuple(
                    int(v)
                    for v in item_payload["metadata"].get(
                        "source_shape",
                        dataset.display_metadata.get("source_shape", dataset.base_shape),
                    )
                ),
                source_spacing=dataset.base_spacing,
                source_affine=item_payload["metadata"].get(
                    "source_affine",
                    dataset.display_metadata.get(
                        "source_affine", dataset.display_metadata.get("affine")
                    ),
                ),
            )

        updated_dataset = dataset
        if compute_result is not None:
            updated_dataset = replace(
                updated_dataset,
                input_state=compute_result.dataset_input,
                layer_names=compute_result.layer_names,
                selected_layer=compute_result.selected_layer,
                feature_size=compute_result.feature_size,
            )
        elif isinstance(result, dict) and isinstance(
            result.get("dataset_input"), DatasetInput
        ):
            updated_dataset = replace(
                updated_dataset,
                input_state=result["dataset_input"],
                layer_names=tuple(str(v) for v in result.get("layer_names", ())),
                selected_layer=str(result.get("selected_layer", dataset.selected_layer)),
                feature_size=int(result.get("feature_size", dataset.feature_size)),
            )
        updated_dataset = updated_dataset.with_result_id(volume.id)
        extra_volumes: list[VolumeRecord] = []
        if compute_result is not None and compute_result.prediction_volume is not None:
            source_prediction = compute_result.prediction_volume
            prediction_id = self.create_prediction_volume_id(
                dataset_id, source_prediction.method_id
            )
            prediction = replace(
                source_prediction,
                id=prediction_id,
                dataset_id=dataset_id,
                display_name=self.unique_volume_display_name(
                    source_prediction.display_name
                ),
                metadata={
                    **source_prediction.metadata,
                    "volume_id": prediction_id,
                },
                source_base_item_id=dataset.base_volume_id,
                source_shape=tuple(
                    int(v)
                    for v in source_prediction.metadata.get(
                        "source_shape",
                        dataset.display_metadata.get("source_shape", dataset.base_shape),
                    )
                ),
                source_spacing=dataset.base_spacing,
                source_affine=source_prediction.metadata.get(
                    "source_affine",
                    dataset.display_metadata.get(
                        "source_affine", dataset.display_metadata.get("affine")
                    ),
                ),
            )
            updated_dataset = updated_dataset.with_result_id(prediction.id)
            extra_volumes.append(prediction)
        self.datasets[dataset_id] = updated_dataset
        self.volumes[volume.id] = volume
        if volume.id not in self.selection.volume_order:
            self.selection = replace(
                self.selection, volume_order=(*self.selection.volume_order, volume.id)
            )
        self.selection = replace(self.selection, selected_volume_id=volume.id)
        self._emit(VolumeUpserted(volume.id, dataset_id))
        for extra_volume in extra_volumes:
            self.volumes[extra_volume.id] = extra_volume
            if extra_volume.id not in self.selection.volume_order:
                self.selection = replace(
                    self.selection,
                    volume_order=(*self.selection.volume_order, extra_volume.id),
                )
            self._emit(VolumeUpserted(extra_volume.id, dataset_id))
        return volume.id

    def create_prediction_volume_id(self, dataset_id: str, method_id: str) -> str:
        return f"{dataset_id}:{method_id}:{uuid4().hex[:6]}"

    def upsert_named_volume(
        self,
        dataset_id: str,
        volume_id: str,
        item_payload: dict[str, Any],
        *,
        emit: bool = True,
    ) -> str | None:
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return None
        existing = self.volumes.get(volume_id)
        display_name = str(item_payload["name"])
        volume = VolumeRecord(
            id=volume_id,
            dataset_id=dataset_id,
            display_name=(existing.display_name if existing is not None else display_name),
            source=str(item_payload["source"]),
            method_id=str(item_payload["method_id"]),
            data=item_payload["data"],
            data_range=item_payload["data_range"],
            transfer_function=item_payload["transfer_function"],
            spacing=tuple(float(v) for v in item_payload["spacing"]),
            metadata={**item_payload["metadata"], "volume_id": volume_id},
            shape=tuple(int(v) for v in item_payload["shape"]),
            source_base_item_id=dataset.base_volume_id,
            source_shape=tuple(
                int(v)
                for v in item_payload["metadata"].get(
                    "source_shape", dataset.display_metadata.get("source_shape", dataset.base_shape)
                )
            ),
            source_spacing=dataset.base_spacing,
            source_affine=item_payload["metadata"].get(
                "source_affine",
                dataset.display_metadata.get("source_affine", dataset.display_metadata.get("affine")),
            ),
            visible=existing.visible if existing is not None else bool(item_payload.get("visible", True)),
            plugin_metadata=dict(item_payload.get("plugin_metadata", {})),
        )
        self.volumes[volume_id] = volume
        self.datasets[dataset_id] = dataset.with_result_id(volume_id)
        if volume_id not in self.selection.volume_order:
            self.selection = replace(
                self.selection, volume_order=(*self.selection.volume_order, volume_id)
            )
        if emit:
            self._emit(VolumeUpserted(volume_id, dataset_id))
        return volume_id

    def notify_volumes_upserted(self, dataset_id: str, volume_ids: list[str]) -> None:
        """Publish one store update after a group of volumes has been inserted."""
        if volume_ids:
            self._emit(VolumeUpserted(volume_ids[-1], dataset_id))

    def unique_volume_display_name(self, base_name: str) -> str:
        existing = {volume.display_name for volume in self.volumes.values()}
        if base_name not in existing:
            return base_name
        suffix = 2
        while f"{base_name}_{suffix}" in existing:
            suffix += 1
        return f"{base_name}_{suffix}"

    def delete_volume(self, volume_id: str) -> bool:
        volume = self.volumes.get(volume_id)
        if volume is None:
            return False
        if volume.source == "base":
            return self.delete_dataset(volume.dataset_id)

        del self.volumes[volume_id]
        dataset = self.datasets.get(volume.dataset_id)
        if dataset is not None:
            self.datasets[volume.dataset_id] = replace(
                dataset,
                result_ids=tuple(
                    item_id for item_id in dataset.result_ids if item_id != volume_id
                ),
            )
        self._remove_volume_from_selection(volume_id)
        self._emit(VolumeDeleted(volume_id, volume.dataset_id))
        self._emit(VolumeOrderChanged())
        self._emit(SelectionChanged(self.selection.selected_volume_id))
        return True

    def delete_dataset(self, dataset_id: str) -> bool:
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return False
        volume_ids = [dataset.base_volume_id, *dataset.result_ids]
        for volume_id in volume_ids:
            self.volumes.pop(volume_id, None)
        del self.datasets[dataset_id]

        active = {
            plugin_id: active_dataset_id
            for plugin_id, active_dataset_id in self.selection.active_dataset_by_plugin.items()
            if active_dataset_id != dataset_id
        }
        next_dataset_id = next(iter(self.datasets), "")
        for plugin_id in ("gradcam", "perturbation"):
            if plugin_id not in active and next_dataset_id:
                active[plugin_id] = next_dataset_id
        volume_order = tuple(
            item_id
            for item_id in self.selection.volume_order
            if item_id not in set(volume_ids)
        )
        selected = (
            self.selection.selected_volume_id
            if self.selection.selected_volume_id in volume_order
            else (volume_order[0] if volume_order else "")
        )
        self.selection = SelectionState(active, selected, volume_order)
        for volume_id in volume_ids:
            self._emit(VolumeDeleted(volume_id, dataset_id))
        self._emit(DatasetDeleted(dataset_id))
        self._emit(VolumeOrderChanged())
        self._emit(SelectionChanged(self.selection.selected_volume_id))
        return True

    def _remove_volume_from_selection(self, volume_id: str) -> None:
        volume_order = tuple(
            item_id for item_id in self.selection.volume_order if item_id != volume_id
        )
        selected = (
            self.selection.selected_volume_id
            if self.selection.selected_volume_id in volume_order
            else (volume_order[0] if volume_order else "")
        )
        self.selection = replace(
            self.selection,
            selected_volume_id=selected,
            volume_order=volume_order,
        )

    def workspace_renderable_items(self) -> list[dict[str, object]]:
        return [
            self.volumes[item_id].to_render_item(
                is_focus=item_id == self.selection.selected_volume_id
            )
            for item_id in self.selection.volume_order
            if item_id in self.volumes
        ]
