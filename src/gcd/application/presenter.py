from __future__ import annotations

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
        self.use_overlay = True
        self.is_recording = False
        self.rotation_speed = 0.5
        self.transfer_function = TransferFunction.overlay_preset()
        self.data_range = DataRange(0.0, 1.0)

        self._connect_signals()

    def _connect_signals(self) -> None:
        self.view.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.view.open_file_button.clicked.connect(self.on_open_file_requested)
        self.view.overlay_button.clicked.connect(self.on_overlay_selected)
        self.view.heatmap_button.clicked.connect(self.on_heatmap_selected)
        self.view.replace_camera_button.clicked.connect(
            self.on_replace_camera_requested
        )
        self.view.speed_slider.valueChanged.connect(self.on_rotation_speed_changed)
        self.view.start_button.clicked.connect(self.on_start_rotation_requested)
        self.view.stop_button.clicked.connect(self.on_stop_rotation_requested)
        self.view.layer_combo.currentTextChanged.connect(self.on_layer_changed)
        self.view.feature_widget.apply_requested.connect(self.on_feature_range_changed)
        self.view.class_spinbox.valueChanged.connect(self.on_class_changed)
        self.view.save_screenshot_button.clicked.connect(
            self.on_save_screenshot_requested
        )
        self.view.record_video_button.clicked.connect(self.on_record_video_requested)
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
        if options:
            self.workflow.set_config(options[0]["path"])
        self.view.set_rotation_speed_label(self.rotation_speed)
        self.view.set_render_mode(self.use_overlay)
        self.view.set_rotation_running(True)
        self.view.renderer.set_rotation_speed(self.rotation_speed)
        self.view.renderer.show_volumes(
            [self.workflow.engine.cam, self.workflow.engine.volume_data],
            [self.workflow.engine.img1_spacing, self.workflow.engine.img1_spacing],
            [
                self.workflow.engine.display_metadata,
                self.workflow.engine.display_metadata,
            ],
        )
        self.view.renderer.start_rotation()
        self.apply_transfer_function_to_renderer()
        self.view.workspace.set_workspace_payload(
            volume_data=self.workflow.engine.volume_data,
            cam_data=self.workflow.engine.cam,
            transfer_function=self.transfer_function,
            data_range=self.data_range,
        )
        self.refresh_roi_panel()
        self.view.transfer_editor.set_transfer_function(
            self.transfer_function, self.data_range
        )

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
            lambda: self.workflow.load_input(file_name, self.view.selected_class()),
            self._on_input_loaded,
            self._on_background_error,
        )

    def _on_input_loaded(self, result: dict) -> None:
        self.current_file = result["file_name"]
        self.view.set_file_name(os.path.basename(self.current_file))
        self.transfer_function = result["transfer_function"]
        self.data_range = result["data_range"]
        self.view.set_layer_options(result["layer_names"], result["selected_layer"])
        self.view.set_feature_size(result["feature_size"])
        self.view.transfer_editor.set_transfer_function(
            self.transfer_function, self.data_range
        )
        self._render_result(result)
        self.view.renderer.store_initial_camera()
        self.view.set_rotation_running(True)

    def _on_background_error(self, exc: Exception) -> None:
        if not getattr(exc, "skip_error_store", False):
            self.error_store.save(exc, context="background_task")
        self.view.set_rotation_running(False)

    def _render_result(self, result: dict) -> None:
        render_request = result["render_request"]
        self.view.renderer.show_volumes(
            render_request["volumes"],
            render_request["spacing"],
            render_request.get("metadata"),
        )
        self.apply_transfer_function_to_renderer()
        self.view.workspace.set_workspace_payload(
            volume_data=self.workflow.engine.volume_data,
            cam_data=self.workflow.engine.cam,
            transfer_function=self.transfer_function,
            data_range=self.data_range,
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
            use_overlay=self.use_overlay,
            transfer_function=self.transfer_function,
        )
        self.data_range = result["data_range"]
        self.view.set_feature_size(result["feature_size"])
        self.view.transfer_editor.set_transfer_function(
            self.transfer_function, self.data_range
        )
        self._render_result(result)

    def on_layer_changed(self, _layer: str) -> None:
        self._compute_and_render_current_selection()

    def on_feature_range_changed(self) -> None:
        self._compute_and_render_current_selection()

    def on_class_changed(self, _value: int) -> None:
        if self.current_file:
            self.task_runner.submit(
                lambda: self.workflow.load_input(
                    self.current_file, self.view.selected_class()
                ),
                self._on_input_loaded,
                self._on_background_error,
            )

    def on_overlay_selected(self) -> None:
        self.use_overlay = True
        self.view.set_render_mode(True)
        self.transfer_function = TransferFunction.overlay_preset()
        self.view.transfer_editor.set_transfer_function(
            self.transfer_function, self.data_range
        )
        self._compute_and_render_current_selection()

    def on_heatmap_selected(self) -> None:
        self.use_overlay = False
        self.view.set_render_mode(False)
        self.transfer_function = TransferFunction.heatmap_preset()
        self.view.transfer_editor.set_transfer_function(
            self.transfer_function, self.data_range
        )
        self._compute_and_render_current_selection()

    def on_transfer_function_changed(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> None:
        self.transfer_function = transfer_function
        self.data_range = data_range
        self.apply_transfer_function_to_renderer()

    def apply_transfer_function_to_renderer(self) -> None:
        colors, opacities = self.transfer_function.renderer_points(self.data_range)
        self.view.renderer.apply_transfer_function(colors, opacities)
        self.view.workspace.set_workspace_payload(
            volume_data=self.workflow.engine.volume_data,
            cam_data=self.workflow.engine.cam,
            transfer_function=self.transfer_function,
            data_range=self.data_range,
        )
        self.refresh_roi_panel()

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
            self.transfer_function, self.data_range = self.transfer_service.load(
                path, canvas_width=width, canvas_height=height
            )
            self.view.transfer_editor.set_transfer_function(
                self.transfer_function, self.data_range
            )
            self.apply_transfer_function_to_renderer()
        except Exception as exc:
            self.error_store.save(exc, context="load_transfer_function")

    def on_save_transfer_requested(self) -> None:
        try:
            path = self.view.transfer_editor.choose_save_path()
            if not path:
                return
            width, height = self.view.transfer_editor.canvas_size()
            self.transfer_service.save(
                path,
                self.transfer_function,
                self.data_range,
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
