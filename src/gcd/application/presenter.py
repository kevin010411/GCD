from __future__ import annotations

import json
import os

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

        self.current_file = ""
        self.is_recording = False
        self.rotation_speed = 0.5
        self.volume_transfer_function = TransferFunction.base_preset()
        self.cam_transfer_function = TransferFunction.heatmap_preset()
        self.volume_data_range = DataRange(0.0, 1.0)
        self.cam_data_range = DataRange(0.0, 1.0)
        self.volume_order = ["volume", "cam"]
        self.volume_visibility = {"volume": True, "cam": True}
        self.selected_transfer_volume_id = "cam"

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
        self.view.transfer_editor.transfer_function_changed.connect(
            self.on_transfer_function_changed
        )
        self.view.transfer_editor.load_requested.connect(
            self.on_load_transfer_requested
        )
        self.view.transfer_editor.save_requested.connect(
            self.on_save_transfer_requested
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
        if options:
            self.workflow.set_config(options[0]["path"])
        self.view.set_rotation_speed_label(self.rotation_speed)
        self.view.set_rotation_running(True)
        self.view.renderer.set_rotation_speed(self.rotation_speed)
        self.view.renderer.show_volumes(
            [self.workflow.engine.volume_data, self.workflow.engine.cam],
            [self.workflow.engine.img1_spacing, self.workflow.engine.img1_spacing],
            [
                self.workflow.engine.display_metadata,
                self.workflow.engine.display_metadata,
            ],
        )
        self.view.renderer.start_rotation()
        self._sync_volume_list()
        self.apply_transfer_function_to_renderer()
        self.view.workspace.set_workspace_payload(
            volume_data=self.workflow.engine.volume_data,
            cam_data=self.workflow.engine.cam,
            volume_transfer_function=self.volume_transfer_function,
            volume_data_range=self.volume_data_range,
            cam_transfer_function=self.cam_transfer_function,
            cam_data_range=self.cam_data_range,
        )
        self._sync_transfer_editor()

    def on_model_changed(self, _index: int) -> None:
        path = self.view.selected_model_path()
        if path:
            self.workflow.set_config(path)

    def on_open_file_requested(self) -> None:
        file_name = self.view.choose_input_file()
        if not file_name:
            return
        self.view.set_file_name(file_name)
        self.current_file = file_name
        self.view.renderer.stop_rotation()
        self.view.set_rotation_running(False)
        self.view.renderer.clear_volumes()
        self.view.renderer.render()
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
        self.current_file = result["file_name"]
        self.view.set_file_name(os.path.basename(self.current_file))
        self.volume_transfer_function = result["volume_transfer_function"]
        self.cam_transfer_function = result["cam_transfer_function"]
        self.volume_data_range = result["volume_data_range"]
        self.cam_data_range = result["cam_data_range"]
        self.view.set_method_options(
            result["method_options"], result["selected_method"]
        )
        self.view.set_layer_options(result["layer_names"], result["selected_layer"])
        self.view.set_feature_size(result["feature_size"])
        self._sync_volume_list()
        self._sync_transfer_editor()
        self._render_result(result)
        self.view.renderer.store_initial_camera()
        self.view.set_rotation_running(True)

    def _on_background_error(self, exc: Exception) -> None:
        if not getattr(exc, "skip_error_store", False):
            self.error_store.save(exc, context="background_task")
        self.view.set_rotation_running(False)

    def _render_result(self, result: dict) -> None:
        self.view.renderer.show_volumes(
            self._ordered_volume_payloads(),
            self._ordered_volume_spacing(),
            self._ordered_volume_metadata(),
        )
        self.apply_transfer_function_to_renderer()
        self.view.workspace.set_workspace_payload(
            volume_data=self.workflow.engine.volume_data,
            cam_data=self.workflow.engine.cam,
            volume_transfer_function=self.volume_transfer_function,
            volume_data_range=self.volume_data_range,
            cam_transfer_function=self.cam_transfer_function,
            cam_data_range=self.cam_data_range,
        )
        self.refresh_roi_panel()

    def _compute_and_render_current_selection(self) -> None:
        if not self.current_file:
            return
        n1, n2 = self.view.feature_range()
        result = self.workflow.compute_cam(
            layer=self.view.selected_layer(),
            n1=n1,
            n2=n2,
            method=self.view.selected_method(),
            cam_transfer_function=self.cam_transfer_function,
            volume_transfer_function=self.volume_transfer_function,
        )
        self.cam_data_range = result["cam_data_range"]
        self.volume_data_range = result["volume_data_range"]
        self.view.set_method_options(
            result["method_options"], result["selected_method"]
        )
        self.view.set_feature_size(result["feature_size"])
        self._sync_volume_list()
        self._sync_transfer_editor()
        self._render_result(result)

    def on_layer_changed(self, _layer: str) -> None:
        self._compute_and_render_current_selection()

    def on_feature_range_changed(self) -> None:
        self._compute_and_render_current_selection()

    def on_class_changed(self, _value: int) -> None:
        if self.current_file:
            self.task_runner.submit(
                lambda: self.workflow.load_input(
                    self.current_file,
                    self.view.selected_class(),
                    self.view.selected_method(),
                ),
                self._on_input_loaded,
                self._on_background_error,
            )

    def on_method_changed(self, _index: int) -> None:
        if self.current_file:
            self.task_runner.submit(
                lambda: self.workflow.load_input(
                    self.current_file,
                    self.view.selected_class(),
                    self.view.selected_method(),
                ),
                self._on_input_loaded,
                self._on_background_error,
            )

    def on_transfer_function_changed(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> None:
        if self._current_transfer_target() == "volume":
            self.volume_transfer_function = transfer_function
        else:
            self.cam_transfer_function = transfer_function
        self.apply_transfer_function_to_renderer()

    def apply_transfer_function_to_renderer(self) -> None:
        cam_colors, cam_opacities = self.cam_transfer_function.renderer_points(self.cam_data_range)
        volume_colors, volume_opacities = self.volume_transfer_function.renderer_points(
            self.volume_data_range
        )
        cam_visible = self._volume_visible("cam")
        volume_visible = self._volume_visible("volume")
        if cam_visible and volume_visible:
            # When both volumes are visible, keep the base anatomy semi-transparent
            # so the heatmap is not completely occluded by the higher-density shell.
            volume_opacities = [
                (value, min(1.0, max(0.0, opacity * 0.35)))
                for value, opacity in volume_opacities
            ]
            cam_opacities = [
                (value, min(1.0, max(0.0, opacity * 1.1)))
                for value, opacity in cam_opacities
            ]
        for index, volume_id in enumerate(self.volume_order):
            if volume_id == "volume":
                colors = volume_colors
                opacities = volume_opacities
            else:
                colors = cam_colors
                opacities = cam_opacities
            self.view.renderer.set_volume_transfer_functions(
                index,
                colors,
                opacities,
                visible=self._volume_visible(volume_id),
                render=False,
            )
        self.view.renderer.render()
        self.view.workspace.set_workspace_payload(
            volume_data=self.workflow.engine.volume_data,
            cam_data=self.workflow.engine.cam,
            volume_transfer_function=self.volume_transfer_function,
            volume_data_range=self.volume_data_range,
            cam_transfer_function=self.cam_transfer_function,
            cam_data_range=self.cam_data_range,
        )
        self.refresh_roi_panel()

    def _current_transfer_target(self) -> str:
        return self.selected_transfer_volume_id or "cam"

    def _current_transfer_state(self) -> tuple[TransferFunction, DataRange]:
        if self._current_transfer_target() == "volume":
            return self.volume_transfer_function, self.volume_data_range
        return self.cam_transfer_function, self.cam_data_range

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
        return "Base Volume" if volume_id == "volume" else "Heatmap Volume"

    def _ordered_volume_payloads(self) -> list[object]:
        return [
            self.workflow.engine.volume_data if volume_id == "volume" else self.workflow.engine.cam
            for volume_id in self.volume_order
        ]

    def _ordered_volume_spacing(self) -> list[tuple[float, float, float]]:
        return [self.workflow.engine.img1_spacing for _ in self.volume_order]

    def _ordered_volume_metadata(self) -> list[dict[str, object]]:
        return [
            {
                **self.workflow.engine.display_metadata,
                "volume_id": volume_id,
            }
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
        self.view.renderer.show_volumes(
            self._ordered_volume_payloads(),
            self._ordered_volume_spacing(),
            self._ordered_volume_metadata(),
        )
        self.apply_transfer_function_to_renderer()

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
        self.view.renderer.replace_camera()

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
            if self._current_transfer_target() == "volume":
                self.volume_transfer_function = transfer_function
            else:
                self.cam_transfer_function = transfer_function
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
