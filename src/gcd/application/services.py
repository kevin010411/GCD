from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import ControlPoint, DataRange, TransferFunction
from ..presentation.qt.workspace_models import (
    AnnotationMode,
    AnnotationState,
    Box2DAnnotation,
    Box3DAnnotation,
    PointAnnotation,
    SliceOrientation,
)


class TransferFunctionAppService:
    def serialize(
        self,
        transfer_function: TransferFunction,
        data_range: DataRange,
        canvas_width: float,
        canvas_height: float,
        *,
        include_alpha: bool = False,
    ) -> dict[str, Any]:
        control_points = []
        for point in transfer_function.control_points:
            x = point.position * canvas_width
            color = point.color
            if include_alpha and len(color) == 7:
                color = f"{color}FF"
            control_points.append(
                {
                    "x": float(x),
                    "color": color,
                    "opacity": float(point.opacity),
                }
            )
        return {
            "version": 1,
            "canvas": {"width": float(canvas_width), "height": float(canvas_height)},
            "data_range": {
                "min": float(data_range.min_value),
                "max": float(data_range.max_value),
            },
            "control_points": control_points,
        }

    def deserialize(
        self, payload: dict[str, Any], *, canvas_width: float, canvas_height: float
    ) -> tuple[TransferFunction, DataRange]:
        if int(payload.get("version", 1)) != 1:
            raise ValueError("Unsupported transfer function version.")
        source_canvas = payload.get("canvas", {})
        source_width = float(source_canvas.get("width", 400.0) or 400.0)
        data_range_payload = payload.get("data_range", {})
        data_range = DataRange(
            float(data_range_payload.get("min", 0.0)),
            float(data_range_payload.get("max", 1.0)),
        )
        control_points = []
        for item in payload.get("control_points", []):
            x = float(item.get("x", 0.0))
            opacity = float(item.get("opacity", 1.0))
            position = (x / max(source_width, 1.0)) if source_width else 0.0
            color = str(item.get("color", "#808080")).strip()
            if len(color) == 9:
                color = color[:7]
            if not color.startswith("#"):
                color = f"#{color}"
            control_points.append(ControlPoint(position, color.upper(), opacity))
        if not control_points:
            raise ValueError("Transfer function JSON does not contain control points.")
        return TransferFunction.from_iterable(control_points), data_range

    def load(
        self, path: str, *, canvas_width: float, canvas_height: float
    ) -> tuple[TransferFunction, DataRange]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self.deserialize(
            payload, canvas_width=canvas_width, canvas_height=canvas_height
        )

    def save(
        self,
        path: str,
        transfer_function: TransferFunction,
        data_range: DataRange,
        *,
        canvas_width: float,
        canvas_height: float,
    ) -> None:
        payload = self.serialize(
            transfer_function,
            data_range,
            canvas_width,
            canvas_height,
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


class WorkflowService:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.transfer_function_service = TransferFunctionAppService()

    def list_model_configs(self) -> list[dict[str, str]]:
        root = Path("src/config/model")
        if not root.exists():
            return []
        return [
            {"name": path.stem, "path": str(path)}
            for path in sorted(root.glob("*.py"))
            if path.is_file() and path.name != "__init__.py"
        ]

    def set_config(self, config_path: str) -> None:
        self.engine.set_config(config_path)

    def list_cam_methods(self) -> list[dict[str, str]]:
        return self.engine.available_cam_methods("grad")

    def list_perturbation_methods(self) -> list[dict[str, str]]:
        return self.engine.available_cam_methods("perturbation")

    def load_input(
        self, file_name: str, target_class: int, method: str | None = None
    ) -> dict[str, Any]:
        self.engine.set_target_class(target_class)
        messages = self.engine.load_volume(file_name)
        return {
            "file_name": file_name,
            "dataset_state": self.engine.export_state(),
            "layer_names": list(self.engine.layers.keys()),
            "selected_layer": self.engine.cfg["default_layer"],
            "method_options": self.list_cam_methods(),
            "selected_method": method or self.engine.active_method_id,
            "feature_size": self.engine.default_feature_size(),
            "volume_data": self.engine.volume_data,
            "spacing": self.engine.img1_spacing,
            "display_metadata": dict(self.engine.display_metadata),
            "volume_data_range": DataRange.from_data([self.engine.volume_data], method="minmax"),
            "volume_transfer_function": TransferFunction.base_preset(),
            "messages": messages,
        }

    def compute_dataset_result(
        self,
        dataset_state: dict[str, Any],
        *,
        target_class: int,
        layer: str | None,
        n1: int,
        n2: int,
        method: str,
        result_name: str,
        method_params: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        self.engine.restore_state(dataset_state)
        self.engine.set_target_class(target_class)
        cfg_name = str(getattr(self.engine.cfg, "filename", "") or "")
        desired_cache_key = f"{cfg_name}|{target_class}|{method}"
        if self._needs_xai_prepare(
            dataset_state,
            desired_cache_key=desired_cache_key,
            method=method,
            layer=layer,
        ):
            self.engine.prepare_xai_inputs(method=method)
        selected_layer = self.engine.compute_cam(
            layer=layer,
            n1=n1,
            n2=n2,
            method=method,
            method_params=method_params,
        )
        cam_data_range = DataRange.from_data([self.engine.cam], method="minmax")
        volume_data_range = DataRange.from_data([self.engine.volume_data], method="minmax")
        default_transfer = (
            TransferFunction.heatmap_preset()
            if method.startswith("grad")
            else TransferFunction.overlay_preset()
        )
        return {
            "dataset_state": self.engine.export_state(),
            "layer_names": list(self.engine.layers.keys()),
            "selected_layer": selected_layer,
            "method_options": self.engine.available_cam_methods(
                "perturbation" if method.startswith("perturb") else "grad"
            ),
            "selected_method": self.engine.active_method_id,
            "feature_size": self.engine.layers[selected_layer],
            "volume_data_range": volume_data_range,
            "renderable_item": {
                "name": result_name,
                "source": "xai",
                "method_id": method,
                "data": self.engine.cam,
                "data_range": cam_data_range,
                "transfer_function": default_transfer,
                "spacing": self.engine.img1_spacing,
                "metadata": {
                    **self.engine.display_metadata,
                },
                "shape": tuple(int(v) for v in self.engine.cam.shape),
            },
        }

    def _needs_xai_prepare(
        self,
        dataset_state: dict[str, Any],
        *,
        desired_cache_key: str,
        method: str,
        layer: str | None,
    ) -> bool:
        if str(dataset_state.get("xai_cache_key", "")) != desired_cache_key:
            return True
        patch = getattr(self.engine, "patch", None)
        if not patch:
            return True
        if any(not isinstance(item, dict) or item.get("method") != method for item in patch):
            return True
        layers = getattr(self.engine, "layers", {}) or {}
        default_layer = str(self.engine.cfg["default_layer"])
        if (
            len(layers) == 1
            and default_layer in layers
            and int(layers.get(default_layer, 0) or 0) <= 1
        ):
            return True
        if layer and layer not in layers:
            return True
        return False

    def compute_cam(
        self,
        *,
        layer: str | None,
        n1: int,
        n2: int,
        method: str | None = None,
        cam_transfer_function: TransferFunction | None = None,
        volume_transfer_function: TransferFunction | None = None,
    ) -> dict[str, Any]:
        selected_layer = self.engine.compute_cam(
            layer=layer,
            n1=n1,
            n2=n2,
            method=method,
        )
        cam_data_range = DataRange.from_data([self.engine.cam], method="minmax")
        volume_data_range = DataRange.from_data(
            [self.engine.volume_data], method="minmax"
        )
        return {
            "layer_names": list(self.engine.layers.keys()),
            "selected_layer": selected_layer,
            "method_options": self.list_cam_methods(),
            "selected_method": self.engine.active_method_id,
            "feature_size": self.engine.layers[selected_layer],
            "render_request": {
                "volumes": [self.engine.volume_data, self.engine.cam],
                "spacing": [self.engine.img1_spacing, self.engine.img1_spacing],
                "metadata": [
                    self.engine.display_metadata,
                    self.engine.display_metadata,
                ],
            },
            "cam_data_range": cam_data_range,
            "volume_data_range": volume_data_range,
            "cam_transfer_function": cam_transfer_function
            or TransferFunction.heatmap_preset(),
            "volume_transfer_function": volume_transfer_function
            or TransferFunction.base_preset(),
        }


class AnnotationJsonService:
    def serialize(self, state: AnnotationState) -> dict[str, Any]:
        return {
            "version": 1,
            "mode": state.mode.value,
            "point_size": int(state.point_size),
            "active_roi_box_id": state.active_roi_box_id,
            "selected_annotation_id": state.selected_annotation_id,
            "points": [
                {
                    "id": item.id,
                    "space": item.space,
                    "position": [float(v) for v in item.position],
                    "size": int(item.size),
                    "source_viewer_id": item.source_viewer_id,
                }
                for item in state.points
            ],
            "boxes_2d": [
                {
                    "id": item.id,
                    "orientation": item.orientation.value,
                    "slice_index": int(item.slice_index),
                    "rect": [float(v) for v in item.rect],
                    "source_viewer_id": item.source_viewer_id,
                }
                for item in state.boxes_2d
            ],
            "boxes_3d": [
                {
                    "id": item.id,
                    "min_corner": [float(v) for v in item.min_corner],
                    "max_corner": [float(v) for v in item.max_corner],
                    "is_roi_target": bool(item.is_roi_target),
                }
                for item in state.boxes_3d
            ],
        }

    def deserialize(self, payload: dict[str, Any]) -> AnnotationState:
        if int(payload.get("version", 1)) != 1:
            raise ValueError("Unsupported annotation JSON version.")
        state = AnnotationState(
            mode=AnnotationMode(str(payload.get("mode", AnnotationMode.OFF.value))),
            point_size=max(1, int(payload.get("point_size", 8))),
            active_roi_box_id=payload.get("active_roi_box_id"),
            selected_annotation_id=payload.get("selected_annotation_id"),
            points=[
                PointAnnotation(
                    id=str(item["id"]),
                    space=str(item.get("space", "voxel")),
                    position=tuple(float(v) for v in item.get("position", [0, 0, 0])),
                    size=max(1, int(item.get("size", 8))),
                    source_viewer_id=str(item.get("source_viewer_id", "")),
                )
                for item in payload.get("points", [])
            ],
            boxes_2d=[
                Box2DAnnotation(
                    id=str(item["id"]),
                    orientation=SliceOrientation(
                        str(item.get("orientation", SliceOrientation.AXIAL.value))
                    ),
                    slice_index=int(item.get("slice_index", 0)),
                    rect=tuple(float(v) for v in item.get("rect", [0, 0, 0, 0])),
                    source_viewer_id=str(item.get("source_viewer_id", "")),
                )
                for item in payload.get("boxes_2d", [])
            ],
            boxes_3d=[
                Box3DAnnotation(
                    id=str(item["id"]),
                    min_corner=tuple(float(v) for v in item.get("min_corner", [0, 0, 0])),
                    max_corner=tuple(float(v) for v in item.get("max_corner", [0, 0, 0])),
                    is_roi_target=bool(item.get("is_roi_target", False)),
                )
                for item in payload.get("boxes_3d", [])
            ],
        )
        return state

    def load(self, path: str) -> AnnotationState:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self.deserialize(payload)

    def save(self, path: str, state: AnnotationState) -> None:
        payload = self.serialize(state)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
