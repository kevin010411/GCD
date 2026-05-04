from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain import ControlPoint, DataRange, TransferFunction


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
            y = (1.0 - point.opacity) * canvas_height
            color = point.color
            if include_alpha and len(color) == 7:
                color = f"{color}FF"
            control_points.append(
                {
                    "x": float(x),
                    "y": float(y),
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
        source_height = float(source_canvas.get("height", 200.0) or 200.0)
        data_range_payload = payload.get("data_range", {})
        data_range = DataRange(
            float(data_range_payload.get("min", 0.0)),
            float(data_range_payload.get("max", 1.0)),
        )
        control_points = []
        for item in payload.get("control_points", []):
            x = float(item.get("x", 0.0))
            opacity = float(item.get("opacity", 1.0))
            if "opacity" not in item:
                y = float(item.get("y", source_height))
                opacity = 1.0 - (y / max(source_height, 1.0))
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

    def load_input(self, file_name: str, target_class: int) -> dict[str, Any]:
        self.engine.set_target_class(target_class)
        messages = self.engine.load_and_process_input(file_name)
        transfer_function = TransferFunction.overlay_preset()
        compute_result = self.compute_cam(
            layer=None,
            n1=0,
            n2=self.engine.default_feature_size(),
            use_overlay=True,
            transfer_function=transfer_function,
        )
        return {
            "file_name": file_name,
            "layer_names": compute_result["layer_names"],
            "selected_layer": compute_result["selected_layer"],
            "feature_size": compute_result["feature_size"],
            "render_request": compute_result["render_request"],
            "data_range": compute_result["data_range"],
            "transfer_function": compute_result["transfer_function"],
            "messages": messages,
        }

    def compute_cam(
        self,
        *,
        layer: str | None,
        n1: int,
        n2: int,
        use_overlay: bool,
        transfer_function: TransferFunction | None = None,
    ) -> dict[str, Any]:
        selected_layer = self.engine.compute_cam(
            layer=layer,
            n1=n1,
            n2=n2,
            use_overlay=use_overlay,
        )
        data_range = DataRange.from_data(
            [self.engine.cam, self.engine.volume_data],
            method="minmax",
            low_q=1.0,
            high_q=100.0,
        )
        return {
            "layer_names": list(self.engine.layers.keys()),
            "selected_layer": selected_layer,
            "feature_size": self.engine.layers[selected_layer],
            "render_request": {
                "volumes": [self.engine.cam, self.engine.volume_data],
                "spacing": [self.engine.img1_spacing, self.engine.img1_spacing],
            },
            "data_range": data_range,
            "transfer_function": transfer_function
            or (
                TransferFunction.overlay_preset()
                if use_overlay
                else TransferFunction.heatmap_preset()
            ),
            "use_overlay": use_overlay,
        }
