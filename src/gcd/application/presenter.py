from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from ..domain import DataRange, TransferFunction


class MainWindowPresenter:
    def __init__(
        self,
        view,
        workflow_service,
        transfer_service,
        annotation_service,
        task_runner,
        error_store,
    ) -> None:
        self.view = view
        self.workflow = workflow_service
        self.transfer_service = transfer_service
        self.annotation_service = annotation_service
        self.task_runner = task_runner
        self.error_store = error_store

        self.is_recording = False
        self.rotation_speed = 0.5
        self.datasets: dict[str, dict[str, object]] = {}
        self.dataset_order: list[str] = []
        self.render_items: dict[str, dict[str, object]] = {}
        self.volume_order: list[str] = []
        self.volume_visibility: dict[str, bool] = {}
        self.selected_transfer_volume_id = ""
        self.selected_grad_dataset_id = ""
        self.selected_perturbation_dataset_id = ""

        self._connect_signals()

    def _connect_signals(self) -> None:
        self.view.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.view.open_file_button.clicked.connect(self.on_open_file_requested)
        self.view.replace_camera_button.clicked.connect(
            self.on_replace_camera_requested
        )
        self.view.speed_slider.valueChanged.connect(self.on_rotation_speed_changed)
        self.view.start_button.clicked.connect(self.on_start_rotation_requested)
        self.view.stop_button.clicked.connect(self.on_stop_rotation_requested)
        self.view.import_camera_button.clicked.connect(
            self.on_import_camera_requested
        )
        self.view.export_camera_button.clicked.connect(
            self.on_export_camera_requested
        )
        self.view.layer_combo.currentTextChanged.connect(self.on_layer_changed)
        self.view.method_combo.currentIndexChanged.connect(self.on_method_changed)
        self.view.feature_widget.apply_requested.connect(self.on_feature_range_changed)
        self.view.class_spinbox.valueChanged.connect(self.on_class_changed)
        self.view.gradcam_dataset_combo.currentIndexChanged.connect(
            self.on_gradcam_dataset_changed
        )
        self.view.gradcam_run_button.clicked.connect(self.on_gradcam_run_requested)
        self.view.perturbation_dataset_combo.currentIndexChanged.connect(
            self.on_perturbation_dataset_changed
        )
        self.view.perturbation_run_button.clicked.connect(
            self.on_perturbation_run_requested
        )
        self.view.save_screenshot_button.clicked.connect(
            self.on_save_screenshot_requested
        )
        self.view.record_video_button.clicked.connect(self.on_record_video_requested)
        self.view.volume_list.selection_changed.connect(
            self.on_transfer_volume_selected
        )
        self.view.volume_list.visibility_changed.connect(
            self.on_volume_visibility_changed
        )
        self.view.volume_list.order_changed.connect(self.on_volume_order_changed)
        self.view.volume_list.name_changed.connect(self.on_transfer_item_renamed)
        self.view.transfer_editor.transfer_function_changed.connect(
            self.on_transfer_function_changed
        )
        self.view.transfer_editor.load_requested.connect(
            self.on_load_transfer_requested
        )
        self.view.transfer_editor.save_requested.connect(
            self.on_save_transfer_requested
        )
        self.view.transfer_editor.export_png_requested.connect(
            self.on_export_transfer_png_requested
        )
        self.view.layout_action_focus.triggered.connect(
            lambda: self.view.workspace.apply_layout("focus_3d")
        )
        self.view.layout_action_triple.triggered.connect(
            lambda: self.view.workspace.apply_layout("triple_slice")
        )
        self.view.layout_action_quad.triggered.connect(
            lambda: self.view.workspace.apply_layout("quad")
        )
        self.view.layout_action_compare.triggered.connect(
            lambda: self.view.workspace.apply_layout("compare")
        )
        self.view.gradcam_plugin_button.clicked.connect(
            lambda: self.view.set_active_plugin("gradcam")
        )
        self.view.transfer_plugin_button.clicked.connect(
            lambda: self.view.set_active_plugin("transfer")
        )
        self.view.camera_plugin_button.clicked.connect(
            lambda: self.view.set_active_plugin("camera")
        )
        self.view.perturbation_plugin_button.clicked.connect(
            lambda: self.view.set_active_plugin("perturbation")
        )
        self.view.roi_plugin_button.clicked.connect(
            lambda: self.view.set_active_plugin("roi")
        )
        self.view.roi_mode_combo.currentIndexChanged.connect(self.on_roi_mode_changed)
        self.view.roi_point_size_slider.valueChanged.connect(
            self.on_roi_point_size_slider_changed
        )
        self.view.roi_point_size_spinbox.valueChanged.connect(
            self.on_roi_point_size_spinbox_changed
        )
        self.view.roi_box_combo.currentIndexChanged.connect(self.on_roi_box_changed)
        self.view.roi_export_button.clicked.connect(self.on_roi_export_requested)
        self.view.roi_import_button.clicked.connect(self.on_roi_import_requested)
        self.view.roi_clear_all_button.clicked.connect(self.on_roi_clear_all_requested)
        self.view.roi_delete_selected_button.clicked.connect(
            self.on_roi_delete_selected_requested
        )
        self.view.roi_annotation_list.currentTextChanged.connect(
            self.on_roi_annotation_selected
        )
        self.view.workspace.annotations_changed.connect(self.refresh_roi_panel)
        self.view.inspector_toggle_button.toggled.connect(self.view.toggle_inspector)

    def initialize(self) -> None:
        options = self.workflow.list_model_configs()
        self.view.set_model_options(options)
        self.view.set_method_options(self.workflow.list_cam_methods(), "gradcam")
        self.view.set_perturbation_method_options(
            self.workflow.list_perturbation_methods(),
            "perturb_occlusion",
        )
        self.view.set_gradcam_dataset_options([], None)
        self.view.set_perturbation_dataset_options([], None)
        if options:
            self.workflow.set_config(options[0]["path"])
        self.view.set_rotation_speed_label(self.rotation_speed)
        self.view.set_rotation_running(True)
        self.view.renderer.set_rotation_speed(self.rotation_speed)
        self.view.renderer.show_volumes([], [], [])
        self.view.renderer.start_rotation()
        self._sync_volume_list()
        self.apply_transfer_function_to_renderer()
        self.view.workspace.set_workspace_payload(renderable_items=[])
        self.view.set_overlay_status_message("")
        self._sync_transfer_editor()

    def on_model_changed(self, _index: int) -> None:
        path = self.view.selected_model_path()
        if path:
            self.workflow.set_config(path)

    def on_open_file_requested(self) -> None:
        file_name = self.view.choose_input_file()
        if not file_name:
            return
        self.view.renderer.stop_rotation()
        self.view.set_rotation_running(False)
        self.task_runner.submit(
            lambda: self.workflow.load_input(
                file_name,
                self.view.selected_class(),
                self.view.selected_method(),
            ),
            self._on_input_loaded,
            self._on_background_error,
        )

    def _on_input_loaded(self, result: dict) -> None:
        dataset_id = f"dataset-{uuid4().hex[:8]}"
        dataset_name = Path(str(result["file_name"])).stem
        base_item_id = f"{dataset_id}:base"
        self.datasets[dataset_id] = {
            "id": dataset_id,
            "name": dataset_name,
            "file_name": result["file_name"],
            "engine_state": result["dataset_state"],
            "layer_names": list(result["layer_names"]),
            "selected_layer": result["selected_layer"],
            "feature_size": result["feature_size"],
            "base_item_id": base_item_id,
            "result_ids": [],
            "base_shape": tuple(int(v) for v in result["volume_data"].shape),
            "base_spacing": tuple(float(v) for v in result["spacing"]),
            "display_metadata": dict(result["display_metadata"]),
            "xai_cache_key": str(result["dataset_state"].get("xai_cache_key", "")),
        }
        self.dataset_order.append(dataset_id)
        self.render_items[base_item_id] = {
            "id": base_item_id,
            "dataset_id": dataset_id,
            "display_name": dataset_name,
            "source": "base",
            "method_id": "base",
            "data": result["volume_data"],
            "data_range": result["volume_data_range"],
            "transfer_function": result["volume_transfer_function"],
            "spacing": result["spacing"],
            "metadata": {
                **result["display_metadata"],
                "volume_id": base_item_id,
            },
            "shape": tuple(int(v) for v in result["volume_data"].shape),
            "source_base_item_id": base_item_id,
            "source_shape": tuple(int(v) for v in result["volume_data"].shape),
            "source_spacing": tuple(float(v) for v in result["spacing"]),
            "source_affine": result["display_metadata"].get("affine"),
        }
        self.volume_order.append(base_item_id)
        self.volume_visibility[base_item_id] = True
        self.selected_grad_dataset_id = dataset_id
        self.selected_perturbation_dataset_id = dataset_id
        if not self.selected_transfer_volume_id:
            self.selected_transfer_volume_id = base_item_id
        self._sync_dataset_controls()
        self._sync_gradcam_controls()
        self._sync_volume_list()
        self._sync_transfer_editor()
        self._render_current_items()
        self.view.workspace.sync_camera_to_visible_volumes()
        self.view.workspace.store_initial_camera()
        self.view.set_rotation_running(True)

    def _on_background_error(self, exc: Exception) -> None:
        if not getattr(exc, "skip_error_store", False):
            self.error_store.save(exc, context="background_task")
        self.view.set_rotation_running(False)

    def _render_current_items(self) -> None:
        self.view.workspace.show_volumes(
            self._ordered_volume_payloads(),
            self._ordered_volume_spacing(),
            self._ordered_volume_metadata(),
        )
        self.apply_transfer_function_to_renderer()
        self.refresh_roi_panel()

    def _run_dataset_method(
        self,
        dataset_id: str,
        *,
        target_class: int,
        layer: str | None,
        n1: int,
        n2: int,
        method: str,
        result_name: str,
        method_params: dict[str, object] | None = None,
        on_success=None,
    ) -> None:
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return
        self.task_runner.submit(
            lambda: self.workflow.compute_dataset_result(
                dataset["engine_state"],
                target_class=target_class,
                layer=layer,
                n1=n1,
                n2=n2,
                method=method,
                result_name=result_name,
                method_params=method_params,
            ),
            on_success or (lambda result: self._on_xai_result_loaded(dataset_id, result)),
            self._on_background_error,
        )

    def on_layer_changed(self, _layer: str) -> None:
        return

    def on_feature_range_changed(self) -> None:
        return

    def on_class_changed(self, _value: int) -> None:
        return

    def on_method_changed(self, _index: int) -> None:
        return

    def on_transfer_function_changed(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> None:
        current = self.render_items.get(self._current_transfer_target())
        if current is None:
            return
        current["transfer_function"] = transfer_function
        current["data_range"] = data_range
        self.apply_transfer_function_to_renderer()

    def apply_transfer_function_to_renderer(self) -> None:
        for index, volume_id in enumerate(self.volume_order):
            item = self.render_items.get(volume_id)
            if item is None:
                continue
            transfer_function = item["transfer_function"]
            data_range = item["data_range"]
            colors, opacities = transfer_function.renderer_points(data_range)
            self.view.workspace.set_volume_transfer_functions(
                index,
                colors,
                opacities,
                visible=self._volume_visible(volume_id),
                render=False,
            )
        self.view.workspace.render()
        self.view.workspace.set_workspace_payload(
            renderable_items=self._workspace_renderable_items()
        )
        self.view.set_overlay_status_message(self.view.workspace.overlay_status_message())
        self.refresh_roi_panel()

    def _current_transfer_target(self) -> str:
        return self.selected_transfer_volume_id

    def _current_transfer_state(self) -> tuple[TransferFunction, DataRange]:
        current = self.render_items.get(self._current_transfer_target())
        if current is None:
            return TransferFunction.base_preset(), DataRange(0.0, 1.0)
        return current["transfer_function"], current["data_range"]

    def _sync_transfer_editor(self) -> None:
        transfer_function, data_range = self._current_transfer_state()
        self.view.transfer_editor.blockSignals(True)
        self.view.transfer_editor.set_transfer_function(transfer_function, data_range)
        self.view.transfer_editor.blockSignals(False)

    def _sync_volume_list(self) -> None:
        self.view.volume_list.set_volumes(
            [
                {
                    "id": volume_id,
                    "display_name": self._volume_display_name(volume_id),
                    "visible": self._volume_visible(volume_id),
                }
                for volume_id in self.volume_order
            ],
            self.selected_transfer_volume_id,
        )

    def _volume_visible(self, volume_id: str) -> bool:
        return bool(self.volume_visibility.get(volume_id, True))

    def _volume_display_name(self, volume_id: str) -> str:
        item = self.render_items.get(volume_id)
        if item is None:
            return volume_id
        return str(item["display_name"])

    def _ordered_volume_payloads(self) -> list[object]:
        return [self.render_items[volume_id]["data"] for volume_id in self.volume_order]

    def _ordered_volume_spacing(self) -> list[tuple[float, float, float]]:
        return [self.render_items[volume_id]["spacing"] for volume_id in self.volume_order]

    def _ordered_volume_metadata(self) -> list[dict[str, object]]:
        return [
            dict(self.render_items[volume_id]["metadata"])
            for volume_id in self.volume_order
        ]

    def on_transfer_volume_selected(self, volume_id: str) -> None:
        self.selected_transfer_volume_id = volume_id
        self._sync_transfer_editor()

    def on_volume_visibility_changed(self, volume_id: str, visible: bool) -> None:
        self.volume_visibility[volume_id] = bool(visible)
        self._sync_volume_list()
        self.apply_transfer_function_to_renderer()

    def on_volume_order_changed(self, ordered_ids: list[str]) -> None:
        if ordered_ids:
            self.volume_order = list(ordered_ids)
        self._sync_volume_list()
        self._render_current_items()

    def on_transfer_item_renamed(self, volume_id: str, name: str) -> None:
        item = self.render_items.get(volume_id)
        if item is None:
            return
        item["display_name"] = name
        self._sync_volume_list()

    def on_gradcam_dataset_changed(self, _index: int) -> None:
        self.selected_grad_dataset_id = self.view.selected_gradcam_dataset()
        self._sync_gradcam_controls()

    def on_perturbation_dataset_changed(self, _index: int) -> None:
        self.selected_perturbation_dataset_id = self.view.selected_perturbation_dataset()

    def on_gradcam_run_requested(self) -> None:
        dataset_id = self.view.selected_gradcam_dataset()
        if not dataset_id:
            return
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return
        method = self.view.selected_method()
        model_name = self._selected_model_name()
        result_name = f"{dataset['name']}_{model_name}_grad方法"
        n1, n2 = self.view.feature_range()
        if int(dataset.get("feature_size", 0) or 0) <= 0 or n2 <= n1:
            n1, n2 = 0, 999
        self._run_dataset_method(
            dataset_id,
            target_class=self.view.selected_class(),
            layer=self.view.selected_layer(),
            n1=n1,
            n2=n2,
            method=method,
            result_name=result_name,
        )

    def on_perturbation_run_requested(self) -> None:
        dataset_id = self.view.selected_perturbation_dataset()
        if not dataset_id:
            return
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return
        method = self.view.selected_perturbation_method()
        model_name = self._selected_model_name()
        result_name = f"{dataset['name']}_{model_name}_perturb方法"
        self._run_dataset_method(
            dataset_id,
            target_class=self.view.perturbation_class_spinbox.value(),
            layer=dataset["selected_layer"],
            n1=0,
            n2=int(dataset["feature_size"]),
            method=method,
            result_name=result_name,
            method_params={
                "block_size": self.view.perturbation_block_size_spinbox.value(),
                "stride": self.view.perturbation_stride_spinbox.value(),
            },
        )

    def _on_xai_result_loaded(self, dataset_id: str, result: dict) -> None:
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return
        dataset["engine_state"] = result["dataset_state"]
        dataset["layer_names"] = list(result["layer_names"])
        dataset["selected_layer"] = result["selected_layer"]
        dataset["feature_size"] = result["feature_size"]
        dataset["xai_cache_key"] = str(result["dataset_state"].get("xai_cache_key", ""))
        item_payload = result["renderable_item"]
        result_id = f"{dataset_id}:{item_payload['method_id']}:{uuid4().hex[:6]}"
        self.render_items[result_id] = {
            "id": result_id,
            "dataset_id": dataset_id,
            "display_name": item_payload["name"],
            "source": item_payload["source"],
            "method_id": item_payload["method_id"],
            "data": item_payload["data"],
            "data_range": item_payload["data_range"],
            "transfer_function": item_payload["transfer_function"],
            "spacing": item_payload["spacing"],
            "metadata": {
                **item_payload["metadata"],
                "volume_id": result_id,
            },
            "shape": tuple(int(v) for v in item_payload["shape"]),
            "source_base_item_id": dataset["base_item_id"],
            "source_shape": tuple(dataset["base_shape"]),
            "source_spacing": tuple(dataset["base_spacing"]),
            "source_affine": dataset["display_metadata"].get("affine"),
        }
        dataset["result_ids"].append(result_id)
        self.volume_order.append(result_id)
        self.volume_visibility[result_id] = True
        self.selected_transfer_volume_id = result_id
        if item_payload["method_id"].startswith("grad"):
            self.view.set_method_options(result["method_options"], result["selected_method"])
            self.view.set_layer_options(result["layer_names"], result["selected_layer"])
            self.view.set_feature_size(result["feature_size"])
        else:
            self.view.set_perturbation_method_options(
                result["method_options"],
                result["selected_method"],
            )
        self._sync_dataset_controls()
        self._sync_gradcam_controls()
        self._sync_volume_list()
        self._sync_transfer_editor()
        self._render_current_items()

    def _sync_dataset_controls(self) -> None:
        options = [
            {"id": dataset_id, "name": str(self.datasets[dataset_id]["name"])}
            for dataset_id in self.dataset_order
        ]
        self.view.set_gradcam_dataset_options(options, self.selected_grad_dataset_id)
        self.view.set_perturbation_dataset_options(
            options, self.selected_perturbation_dataset_id
        )

    def _sync_gradcam_controls(self) -> None:
        dataset = self.datasets.get(self.selected_grad_dataset_id)
        if dataset is None:
            self.view.set_layer_options([], "")
            self.view.set_feature_size(0)
            return
        self.view.set_layer_options(dataset["layer_names"], dataset["selected_layer"])
        self.view.set_feature_size(int(dataset["feature_size"]))

    def _workspace_renderable_items(self) -> list[dict[str, object]]:
        return [
            {
                "id": item_id,
                "dataset_id": self.render_items[item_id]["dataset_id"],
                "display_name": self.render_items[item_id]["display_name"],
                "source": self.render_items[item_id]["source"],
                "method_id": self.render_items[item_id]["method_id"],
                "data": self.render_items[item_id]["data"],
                "data_range": self.render_items[item_id]["data_range"],
                "transfer_function": self.render_items[item_id]["transfer_function"],
                "visible": self._volume_visible(item_id),
                "spacing": self.render_items[item_id]["spacing"],
                "metadata": self.render_items[item_id]["metadata"],
                "shape": self.render_items[item_id]["shape"],
                "source_base_item_id": self.render_items[item_id]["source_base_item_id"],
                "source_shape": self.render_items[item_id]["source_shape"],
                "source_spacing": self.render_items[item_id]["source_spacing"],
                "source_affine": self.render_items[item_id]["source_affine"],
                "is_focus": item_id == self.selected_transfer_volume_id,
            }
            for item_id in self.volume_order
        ]

    def _selected_model_name(self) -> str:
        path = self.view.selected_model_path()
        return Path(path).stem if path else "model"

    def on_rotation_speed_changed(self, value: int) -> None:
        self.rotation_speed = min(value / 10.0, 10.0)
        self.view.renderer.set_rotation_speed(self.rotation_speed)
        self.view.set_rotation_speed_label(self.rotation_speed)
        if self.rotation_speed == 0.0:
            self.view.renderer.stop_rotation()
            self.view.set_rotation_running(False)
        else:
            self.view.renderer.start_rotation()
            self.view.set_rotation_running(True)

    def on_start_rotation_requested(self) -> None:
        self.view.renderer.start_rotation()
        self.view.set_rotation_running(True)

    def on_stop_rotation_requested(self) -> None:
        self.view.renderer.stop_rotation()
        self.view.set_rotation_running(False)

    def on_replace_camera_requested(self) -> None:
        self.view.workspace.replace_camera()

    def on_import_camera_requested(self) -> None:
        try:
            path = self.view.choose_camera_import_file()
            if not path:
                return
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            snapshot = {
                "position": tuple(float(v) for v in payload["position"]),
                "focal_point": tuple(float(v) for v in payload["focal_point"]),
                "view_up": tuple(float(v) for v in payload["view_up"]),
                "parallel_scale": float(payload.get("parallel_scale", 1.0)),
            }
            self.view.renderer.apply_camera_state(snapshot)
        except Exception as exc:
            self.error_store.save(exc, context="import_camera")

    def on_export_camera_requested(self) -> None:
        try:
            path = self.view.choose_camera_export_file()
            if not path:
                return
            snapshot = self.view.renderer.capture_camera_state()
            if snapshot is None:
                return
            payload = {
                "position": [float(v) for v in snapshot["position"]],
                "focal_point": [float(v) for v in snapshot["focal_point"]],
                "view_up": [float(v) for v in snapshot["view_up"]],
                "parallel_scale": float(snapshot["parallel_scale"]),
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.error_store.save(exc, context="export_camera")

    def on_save_screenshot_requested(self) -> None:
        try:
            file_name = self.view.choose_screenshot_file()
            if not file_name:
                return
            self.view.renderer.save_screenshot(file_name)
        except Exception as exc:
            self.error_store.save(exc, context="save_screenshot")

    def on_record_video_requested(self) -> None:
        try:
            if self.is_recording:
                return
            file_name = self.view.choose_video_file()
            if not file_name:
                return
            self.is_recording = True
            was_rotating = self.view.renderer.rotating
            self.view.renderer.stop_rotation()
            self.view.set_rotation_running(False)
            self.view.renderer.record_rotation_video(file_name, self.rotation_speed)
            if was_rotating:
                self.view.renderer.start_rotation()
                self.view.set_rotation_running(True)
        except Exception as exc:
            self.error_store.save(exc, context="record_video")
        finally:
            self.is_recording = False

    def on_load_transfer_requested(self) -> None:
        try:
            path = self.view.transfer_editor.choose_load_path()
            if not path:
                return
            width, height = self.view.transfer_editor.canvas_size()
            transfer_function, _loaded_range = self.transfer_service.load(
                path, canvas_width=width, canvas_height=height
            )
            current = self.render_items.get(self._current_transfer_target())
            if current is None:
                return
            current["transfer_function"] = transfer_function
            self._sync_transfer_editor()
            self.apply_transfer_function_to_renderer()
        except Exception as exc:
            self.error_store.save(exc, context="load_transfer_function")

    def on_save_transfer_requested(self) -> None:
        try:
            path = self.view.transfer_editor.choose_save_path()
            if not path:
                return
            width, height = self.view.transfer_editor.canvas_size()
            transfer_function, data_range = self._current_transfer_state()
            self.transfer_service.save(
                path,
                transfer_function,
                data_range,
                canvas_width=width,
                canvas_height=height,
            )
        except Exception as exc:
            self.error_store.save(exc, context="save_transfer_function")

    def on_export_transfer_png_requested(self) -> None:
        try:
            path = self.view.transfer_editor.choose_export_png_path()
            if not path:
                return
            transfer_function, data_range = self._current_transfer_state()
            self.view.transfer_editor.export_png(path, transfer_function, data_range)
        except Exception as exc:
            self.error_store.save(exc, context="export_transfer_function_png")

    def on_roi_mode_changed(self, _index: int) -> None:
        mode = self.view.roi_mode_combo.currentData()
        if mode:
            self.view.workspace.set_annotation_mode(mode)

    def on_roi_point_size_slider_changed(self, value: int) -> None:
        self.view.roi_point_size_spinbox.blockSignals(True)
        self.view.roi_point_size_spinbox.setValue(value)
        self.view.roi_point_size_spinbox.blockSignals(False)
        self.view.workspace.set_annotation_point_size(value)

    def on_roi_point_size_spinbox_changed(self, value: int) -> None:
        self.view.roi_point_size_slider.blockSignals(True)
        self.view.roi_point_size_slider.setValue(value)
        self.view.roi_point_size_slider.blockSignals(False)
        self.view.workspace.set_annotation_point_size(value)

    def on_roi_box_changed(self, _index: int) -> None:
        self.view.workspace.set_active_roi_box(self.view.roi_box_combo.currentData())

    def on_roi_export_requested(self) -> None:
        try:
            path = self.view.choose_annotation_export_file()
            if not path:
                return
            self.annotation_service.save(path, self.view.workspace.export_annotations())
        except Exception as exc:
            self.error_store.save(exc, context="export_annotations")

    def on_roi_import_requested(self) -> None:
        try:
            path = self.view.choose_annotation_import_file()
            if not path:
                return
            state = self.annotation_service.load(path)
            self.view.workspace.import_annotations(state)
            self.refresh_roi_panel()
        except Exception as exc:
            self.error_store.save(exc, context="import_annotations")

    def on_roi_clear_all_requested(self) -> None:
        self.view.workspace.clear_annotations()
        self.refresh_roi_panel()

    def on_roi_delete_selected_requested(self) -> None:
        self.view.workspace.delete_selected_annotation()
        self.refresh_roi_panel()

    def on_roi_annotation_selected(self, text: str) -> None:
        if not text:
            return
        annotation_id = text.split(" | ", 1)[0]
        self.view.workspace.select_annotation(annotation_id)

    def refresh_roi_panel(self, *_args) -> None:
        state = self.view.workspace.export_annotations()
        self.view.roi_mode_combo.blockSignals(True)
        self.view.roi_mode_combo.setCurrentIndex(
            max(0, self.view.roi_mode_combo.findData(state.mode.value))
        )
        self.view.roi_mode_combo.blockSignals(False)

        self.view.roi_point_size_slider.blockSignals(True)
        self.view.roi_point_size_spinbox.blockSignals(True)
        self.view.roi_point_size_slider.setValue(state.point_size)
        self.view.roi_point_size_spinbox.setValue(state.point_size)
        self.view.roi_point_size_slider.blockSignals(False)
        self.view.roi_point_size_spinbox.blockSignals(False)

        self.view.roi_box_combo.blockSignals(True)
        self.view.roi_box_combo.clear()
        self.view.roi_box_combo.addItem("None", None)
        for box in state.boxes_3d:
            self.view.roi_box_combo.addItem(box.id, box.id)
        self.view.roi_box_combo.setCurrentIndex(
            max(0, self.view.roi_box_combo.findData(state.active_roi_box_id))
        )
        self.view.roi_box_combo.blockSignals(False)

        self.view.roi_annotation_list.blockSignals(True)
        self.view.roi_annotation_list.clear()
        for item in state.points:
            self.view.roi_annotation_list.addItem(f"{item.id} | Point | {item.space}")
        for item in state.boxes_2d:
            self.view.roi_annotation_list.addItem(
                f"{item.id} | 2D Box | {item.orientation.value}@{item.slice_index}"
            )
        for item in state.boxes_3d:
            suffix = " | ROI" if item.id == state.active_roi_box_id else ""
            self.view.roi_annotation_list.addItem(f"{item.id} | 3D Box{suffix}")
        self.view.roi_annotation_list.blockSignals(False)
