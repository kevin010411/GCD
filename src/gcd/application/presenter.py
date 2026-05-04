from __future__ import annotations

import os

from ..domain import DataRange, TransferFunction


class MainWindowPresenter:
    def __init__(
        self, view, workflow_service, transfer_service, task_runner, error_store
    ) -> None:
        self.view = view
        self.workflow = workflow_service
        self.transfer_service = transfer_service
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

    def initialize(self) -> None:
        options = self.workflow.list_model_configs()
        self.view.set_model_options(options)
        if options:
            self.workflow.set_config(options[0]["path"])
        self.view.set_rotation_speed_label(self.rotation_speed)
        self.view.set_rotation_running(True)
        self.view.renderer.set_rotation_speed(self.rotation_speed)
        self.view.renderer.show_volumes(
            [self.workflow.engine.cam, self.workflow.engine.volume_data],
            [self.workflow.engine.img1_spacing, self.workflow.engine.img1_spacing],
        )
        self.view.renderer.start_rotation()
        self.apply_transfer_function_to_renderer()
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
            render_request["volumes"], render_request["spacing"]
        )
        self.apply_transfer_function_to_renderer()

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
        self.transfer_function = TransferFunction.overlay_preset()
        self.view.transfer_editor.set_transfer_function(
            self.transfer_function, self.data_range
        )
        self._compute_and_render_current_selection()

    def on_heatmap_selected(self) -> None:
        self.use_overlay = False
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
