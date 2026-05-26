from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ..domain import DataRange, TransferFunction
from ..domain.workspace_data import WorkspaceEvent, XaiComputeRequest
from .workspace_store import WorkspaceDataStore


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
        self.data_store = WorkspaceDataStore()
        self.selected_grad_dataset_id = ""
        self.selected_perturbation_dataset_id = ""
        self.data_store.subscribe(self._on_store_event)

        self._connect_signals()

    @property
    def datasets(self) -> dict[str, dict[str, object]]:
        return self.data_store.datasets

    @property
    def dataset_order(self) -> list[str]:
        return self.data_store.dataset_order

    @property
    def render_items(self) -> dict[str, dict[str, object]]:
        return self.data_store.render_items

    @property
    def volume_order(self) -> list[str]:
        return self.data_store.volume_order

    @property
    def volume_visibility(self) -> dict[str, bool]:
        return self.data_store.volume_visibility

    @property
    def selected_transfer_volume_id(self) -> str:
        return self.data_store.selected_transfer_volume_id

    @selected_transfer_volume_id.setter
    def selected_transfer_volume_id(self, value: str) -> None:
        self.data_store.selected_transfer_volume_id = value

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
        if hasattr(self.view.volume_list, "delete_requested"):
            self.view.volume_list.delete_requested.connect(self.on_volume_delete_requested)
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
            lambda: self.view.set_active_plugin("data")
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
        self.data_store.add_loaded_dataset(dataset_id, result)
        self.selected_grad_dataset_id = dataset_id
        self.selected_perturbation_dataset_id = dataset_id
        self.view.set_rotation_running(True)

    def _on_store_event(self, event: WorkspaceEvent) -> None:
        if event.name in {"dataset_added", "volume_upserted", "dataset_deleted", "volume_deleted"}:
            self._sync_selected_dataset_ids()
            self._sync_dataset_controls()
            self._sync_gradcam_controls()
            self._sync_volume_list()
            self._sync_transfer_editor()
            self._render_current_items()
        elif event.name in {"volume_order_changed", "transfer_changed"}:
            self._sync_volume_list()
            self._render_current_items()
        elif event.name == "selection_changed":
            self._sync_transfer_editor()
            self._sync_volume_list()
            self.apply_transfer_function_to_renderer()

    def _on_background_error(self, exc: Exception) -> None:
        if not getattr(exc, "skip_error_store", False):
            self.error_store.save(exc, context="background_task")
        self.view.set_rotation_running(False)

    def _render_current_items(self) -> None:
        self.view.workspace.show_volumes(
            self.data_store.ordered_volume_payloads(),
            self.data_store.ordered_volume_spacing(),
            self.data_store.ordered_volume_metadata(),
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
            lambda: self.workflow.compute_xai(
                dataset.input_state,
                XaiComputeRequest(
                    target_class=target_class,
                    layer=layer,
                    n1=n1,
                    n2=n2,
                    method=method,
                    result_name=result_name,
                    method_params=method_params,
                ),
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
        uses_layer_controls = (
            self.view.selected_method_uses_layer_controls()
            if hasattr(self.view, "selected_method_uses_layer_controls")
            else True
        )
        self.view.set_gradcam_layer_controls_enabled(uses_layer_controls)

    def on_transfer_function_changed(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> None:
        if not self.data_store.update_current_transfer_state(
            transfer_function, data_range
        ):
            return

    def apply_transfer_function_to_renderer(self) -> None:
        for index, volume_id in enumerate(self.data_store.volume_order):
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
            renderable_items=self.data_store.workspace_renderable_items()
        )
        self.view.set_overlay_status_message(self.view.workspace.overlay_status_message())
        self.refresh_roi_panel()

    def _current_transfer_target(self) -> str:
        return self.data_store.current_transfer_target()

    def _current_transfer_state(self) -> tuple[TransferFunction, DataRange]:
        return self.data_store.current_transfer_state()

    def _sync_transfer_editor(self) -> None:
        transfer_function, data_range = self._current_transfer_state()
        self.view.transfer_editor.blockSignals(True)
        self.view.transfer_editor.set_transfer_function(transfer_function, data_range)
        self.view.transfer_editor.blockSignals(False)

    def _sync_volume_list(self) -> None:
        self.view.volume_list.set_volumes(
            self.data_store.volume_list_items(),
            self.data_store.selected_transfer_volume_id,
        )

    def _volume_visible(self, volume_id: str) -> bool:
        return self.data_store.volume_visible(volume_id)

    def _volume_display_name(self, volume_id: str) -> str:
        return self.data_store.volume_display_name(volume_id)

    def _ordered_volume_payloads(self) -> list[object]:
        return self.data_store.ordered_volume_payloads()

    def _ordered_volume_spacing(self) -> list[tuple[float, float, float]]:
        return self.data_store.ordered_volume_spacing()

    def _ordered_volume_metadata(self) -> list[dict[str, object]]:
        return self.data_store.ordered_volume_metadata()

    def on_transfer_volume_selected(self, volume_id: str) -> None:
        self.data_store.set_selected_transfer_volume(volume_id)

    def on_volume_visibility_changed(self, volume_id: str, visible: bool) -> None:
        self.data_store.set_volume_visibility(volume_id, visible)

    def on_volume_order_changed(self, ordered_ids: list[str]) -> None:
        self.data_store.set_volume_order(ordered_ids)

    def on_transfer_item_renamed(self, volume_id: str, name: str) -> None:
        if not self.data_store.rename_item(volume_id, name):
            return

    def on_volume_delete_requested(self, volume_id: str) -> None:
        self.data_store.delete_volume(volume_id)

    def on_gradcam_dataset_changed(self, _index: int) -> None:
        self.selected_grad_dataset_id = self.view.selected_gradcam_dataset()
        self.data_store.set_active_dataset("gradcam", self.selected_grad_dataset_id)
        self._sync_gradcam_controls()

    def on_perturbation_dataset_changed(self, _index: int) -> None:
        self.selected_perturbation_dataset_id = self.view.selected_perturbation_dataset()
        self.data_store.set_active_dataset(
            "perturbation", self.selected_perturbation_dataset_id
        )

    def on_gradcam_run_requested(self) -> None:
        dataset_id = self.view.selected_gradcam_dataset()
        if not dataset_id:
            return
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return
        method = self.view.selected_method()
        model_name = self._selected_model_name()
        uses_layer_controls = (
            self.view.selected_method_uses_layer_controls()
            if hasattr(self.view, "selected_method_uses_layer_controls")
            else True
        )
        layer = self.view.selected_layer() if uses_layer_controls else "input"
        target_class = self.view.selected_class()
        result_name = self._prediction_result_name(
            dataset.name,
            model_name,
            layer,
            target_class,
        )
        n1, n2 = self.view.feature_range() if uses_layer_controls else (0, 1)
        if uses_layer_controls and (int(dataset.feature_size or 0) <= 0 or n2 <= n1):
            n1, n2 = 0, 999
        self._run_dataset_method(
            dataset_id,
            target_class=target_class,
            layer=layer,
            n1=n1,
            n2=n2,
            method=method,
            result_name=result_name,
            method_params={
                "model_name": model_name,
                "requested_layer": layer,
                "target_class": target_class,
            },
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
        target_class = self.view.perturbation_class_spinbox.value()
        result_name = self._prediction_result_name(
            dataset.name,
            model_name,
            dataset.selected_layer,
            target_class,
        )
        self._run_dataset_method(
            dataset_id,
            target_class=target_class,
            layer=dataset.selected_layer,
            n1=0,
            n2=int(dataset.feature_size),
            method=method,
            result_name=result_name,
            method_params={
                "block_size": self.view.perturbation_block_size_spinbox.value(),
                "stride": self.view.perturbation_stride_spinbox.value(),
                "model_name": model_name,
                "requested_layer": dataset.selected_layer,
                "target_class": target_class,
            },
        )

    def _on_xai_result_loaded(self, dataset_id: str, result) -> None:
        result_id = self.data_store.upsert_xai_result(dataset_id, result)
        if result_id is None:
            return
        method_id = (
            result.volume.method_id
            if hasattr(result, "volume")
            else str(result["renderable_item"]["method_id"])
        )
        if not method_id.startswith("perturb"):
            self.view.set_method_options(
                list(result.method_options), result.selected_method
            )
            self.view.set_layer_options(list(result.layer_names), result.selected_layer)
            self.view.set_feature_size(result.feature_size)
            self.on_method_changed(0)
        else:
            self.view.set_perturbation_method_options(
                list(result.method_options),
                result.selected_method,
            )

    def _sync_dataset_controls(self) -> None:
        options = self.data_store.dataset_options()
        self.view.set_gradcam_dataset_options(options, self.selected_grad_dataset_id)
        self.view.set_perturbation_dataset_options(
            options, self.selected_perturbation_dataset_id
        )

    def _sync_gradcam_controls(self) -> None:
        dataset = self.datasets.get(self.selected_grad_dataset_id)
        if dataset is None:
            self.view.set_layer_options([], "")
            self.view.set_feature_size(0)
            self.on_method_changed(0)
            return
        self.view.set_layer_options(list(dataset.layer_names), dataset.selected_layer)
        self.view.set_feature_size(int(dataset.feature_size))
        self.on_method_changed(0)

    def _sync_selected_dataset_ids(self) -> None:
        if self.selected_grad_dataset_id not in self.datasets:
            self.selected_grad_dataset_id = self.data_store.active_dataset_id("gradcam")
        if self.selected_perturbation_dataset_id not in self.datasets:
            self.selected_perturbation_dataset_id = self.data_store.active_dataset_id(
                "perturbation"
            )

    def _workspace_renderable_items(self) -> list[dict[str, object]]:
        return self.data_store.workspace_renderable_items()

    def _selected_model_name(self) -> str:
        path = self.view.selected_model_path()
        return Path(path).stem if path else "model"

    @staticmethod
    def _prediction_result_name(
        dataset_name: str,
        model_name: str,
        layer: str | None,
        target_class: int,
    ) -> str:
        selected_layer = str(layer or "layer")
        return f"{dataset_name}_{model_name}_{selected_layer}_class{int(target_class)}"

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
            _current_tf, data_range = self._current_transfer_state()
            if not self.data_store.update_current_transfer_state(
                transfer_function, data_range
            ):
                return
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
