from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import (
    ControlPoint,
    DataRange,
    DatasetInput,
    TransferFunction,
    VolumeRecord,
    XaiComputeRequest,
    XaiComputeResult,
)
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

    def list_current_model_layers(self) -> dict[str, Any]:
        return self.engine.model_layer_metadata()

    def list_cam_methods(self) -> list[dict[str, object]]:
        return self.engine.available_cam_methods("grad")

    def list_xai_method_families(self) -> list[dict[str, object]]:
        if hasattr(self.engine, "available_xai_method_families"):
            return self.engine.available_xai_method_families()
        return [
            {"id": "gradient", "title": "Gradient XAI", "button_label": "Gradient"},
            {
                "id": "perturbation",
                "title": "Perturbation XAI",
                "button_label": "Perturb",
            },
        ]

    def list_xai_methods(self, family: str | None = None) -> list[dict[str, object]]:
        if hasattr(self.engine, "available_xai_methods"):
            return self.engine.available_xai_methods(family)
        if family == "perturbation":
            return self.list_perturbation_methods()
        return self.list_cam_methods()

    def list_objectives(self, family: str | None = None) -> list[dict[str, object]]:
        return self.engine.available_objectives(family)

    def list_perturbation_methods(self) -> list[dict[str, object]]:
        return self.engine.available_cam_methods("perturbation")

    def load_input(
        self, file_name: str, target_class: int, method: str | None = None
    ) -> dict[str, Any]:
        self.engine.set_target_class(target_class)
        messages = self.engine.load_volume(file_name)
        return {
            "file_name": file_name,
            "dataset_input": self.engine.dataset_input(),
            "layer_names": [],
            "selected_layer": "",
            "method_options": self.list_cam_methods(),
            "selected_method": method or self.engine.active_method_id,
            "objective_options": self.list_objectives("gradient"),
            "selected_objective": self.engine.active_objective_id,
            "feature_size": 0,
            "volume_data": self.engine.volume_data,
            "spacing": self.engine.img1_spacing,
            "display_metadata": dict(self.engine.display_metadata),
            "volume_data_range": DataRange.from_data([self.engine.volume_data], method="minmax"),
            "volume_transfer_function": TransferFunction.heatmap_preset(),
            "messages": messages,
        }

    def compute_dataset_result(
        self,
        dataset_input: DatasetInput,
        *,
        target_class: int,
        layer: str | None,
        n1: int,
        n2: int,
        method: str,
        result_name: str,
        objective_id: str = "predicted_target_mask",
        method_params: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        effective_objective_id = (
            "predicted_mask_dice"
            if method.startswith("perturb") and objective_id == "predicted_target_mask"
            else objective_id
        )
        result = self.compute_xai(
            dataset_input,
            XaiComputeRequest(
                target_class=target_class,
                layer=layer,
                n1=n1,
                n2=n2,
                method=method,
                objective_id=effective_objective_id,
                result_name=result_name,
                method_params=method_params,
            ),
        )
        return {
            "dataset_input": result.dataset_input,
            "layer_names": list(result.layer_names),
            "selected_layer": result.selected_layer,
            "method_options": list(result.method_options),
            "selected_method": result.selected_method,
            "objective_options": list(result.objective_options),
            "selected_objective": result.selected_objective,
            "feature_size": result.feature_size,
            "volume_data_range": result.volume_data_range,
            "renderable_item": {
                "name": result.volume.display_name,
                "source": result.volume.source,
                "method_id": result.volume.method_id,
                "data": result.volume.data,
                "data_range": result.volume.data_range,
                "transfer_function": result.volume.transfer_function,
                "spacing": result.volume.spacing,
                "metadata": result.volume.metadata,
                "shape": result.volume.shape,
            },
        }

    def compute_xai(
        self, dataset_input: DatasetInput, request: XaiComputeRequest
    ) -> XaiComputeResult:
        self.engine.load_dataset_input(dataset_input)
        self.engine.set_target_class(request.target_class)
        self.engine.prepare_xai_inputs(
            method=request.method,
            objective_id=request.objective_id,
            method_params=request.method_params,
        )
        selected_layer = self.engine.compute_cam(
            layer=request.layer,
            n1=request.n1,
            n2=request.n2,
            method=request.method,
            method_params=request.method_params,
        )
        cam_data_range = DataRange.from_data([self.engine.cam], method="minmax")
        volume_data_range = DataRange.from_data([self.engine.volume_data], method="minmax")
        default_transfer = TransferFunction.heatmap_preset()
        family = "perturbation" if request.method.startswith("perturb") else "gradient"
        if hasattr(self.engine, "_resolve_cam_method"):
            family = getattr(self.engine._resolve_cam_method(request.method), "family", family)
        method_options = self.list_xai_methods(family)
        objective_options = self.engine.available_objectives(family)
        return XaiComputeResult(
            dataset_input=self.engine.dataset_input(),
            layer_names=tuple(self.engine.layers.keys()),
            selected_layer=selected_layer,
            method_options=tuple(method_options),
            selected_method=self.engine.active_method_id,
            objective_options=tuple(objective_options),
            selected_objective=self.engine.active_objective_id,
            feature_size=int(self.engine.layers[selected_layer]),
            volume=VolumeRecord(
                id="",
                dataset_id="",
                display_name=request.result_name,
                source="xai",
                method_id=request.method,
                data=self.engine.cam,
                data_range=cam_data_range,
                transfer_function=default_transfer,
                spacing=tuple(float(v) for v in self.engine.img1_spacing),
                metadata={**self.engine.display_metadata},
                shape=tuple(int(v) for v in self.engine.cam.shape),
                source_base_item_id="",
                source_shape=tuple(int(v) for v in self.engine.cam.shape),
                source_spacing=tuple(float(v) for v in self.engine.img1_spacing),
                source_affine=self.engine.display_metadata.get("affine"),
                plugin_metadata={
                    "model_name": (request.method_params or {}).get("model_name"),
                    "requested_layer": request.layer,
                    "selected_layer": selected_layer,
                    "target_class": request.target_class,
                    "method_params": dict(request.method_params or {}),
                },
            ),
            volume_data_range=volume_data_range,
        )

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
            "objective_options": self.list_objectives("gradient"),
            "selected_objective": self.engine.active_objective_id,
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
            or TransferFunction.heatmap_preset(),
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
