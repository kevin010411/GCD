from __future__ import annotations

import json
import threading
from inspect import signature
from pathlib import Path
from uuid import uuid4

from ..domain import DataRange, TransferFunction
from ..domain.workspace_data import WorkspaceEvent, XaiComputeRequest
from .services import VolumePersistenceService
from .workspace_store import WorkspaceDataStore


class PerturbationPauseController:
    def __init__(self) -> None:
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._paused = False

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        if paused:
            self._resume_event.clear()
        else:
            self._resume_event.set()

    def wait_if_paused(self) -> None:
        while not self._resume_event.wait(0.01):
            pass

    def is_paused(self) -> bool:
        return self._paused


class MainWindowPresenter:
    def __init__(
        self,
        view,
        workflow_service,
        transfer_service,
        annotation_service,
        task_runner,
        error_store,
        volume_service=None,
    ) -> None:
        self.view = view
        self.workflow = workflow_service
        self.transfer_service = transfer_service
        self.annotation_service = annotation_service
        self.volume_service = volume_service or VolumePersistenceService()
        self.task_runner = task_runner
        self.error_store = error_store

        self.is_recording = False
        self.rotation_speed = 0.5
        self.data_store = WorkspaceDataStore()
        self.selected_grad_dataset_id = ""
        self.selected_perturbation_dataset_id = ""
        self.selected_perturbation_answer_volume_id = ""
        self.selected_xai_dataset_by_family = {"gradient": "", "perturbation": ""}
        self._xai_is_running = False
        self._xai_running_family = ""
        self._perturb_preview_active = False
        self._perturb_preview_enabled = False
        self._perturb_pause_controller: PerturbationPauseController | None = None
        self._pre_perturb_preview_rotation_running = False
        self._last_scene_signature: tuple[tuple[str, tuple[int, ...]], ...] = ()
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
        if hasattr(self.view, "perturbation_method_combo"):
            self.view.perturbation_method_combo.currentIndexChanged.connect(
                lambda _index: self.on_xai_method_changed("perturbation")
            )
        if hasattr(self.view, "perturbation_preview_pause_button"):
            self.view.perturbation_preview_pause_button.toggled.connect(
                self.on_perturbation_preview_pause_toggled
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
        if hasattr(self.view.volume_list, "save_requested"):
            self.view.volume_list.save_requested.connect(self.on_volume_save_requested)
        self.view.transfer_editor.transfer_function_changed.connect(
            self.on_transfer_function_changed
        )
        if hasattr(self.view.transfer_editor, "transfer_function_change_finished"):
            self.view.transfer_editor.transfer_function_change_finished.connect(
                self.on_transfer_function_change_finished
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
        self.view.layout_action_3d_only.triggered.connect(
            lambda: self.view.workspace.apply_layout("3d_only")
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
        if hasattr(self.workflow, "list_xai_methods") and hasattr(
            self.view, "set_xai_method_options"
        ):
            self.view.set_xai_method_options(
                "gradient", self.workflow.list_xai_methods("gradient"), "gradcam"
            )
            self.view.set_xai_method_options(
                "perturbation",
                self.workflow.list_xai_methods("perturbation"),
                "perturb_occlusion",
            )
        else:
            self.view.set_method_options(self.workflow.list_cam_methods(), "gradcam")
        if hasattr(self.view, "set_xai_objective_options"):
            self.view.set_xai_objective_options(
                "gradient",
                self._list_objectives("gradient"),
                "predicted_target_mask",
            )
            self.view.set_xai_objective_options(
                "perturbation",
                self._list_objectives("perturbation"),
                "predicted_mask_dice",
            )
        else:
            self.view.set_objective_options(
                self._list_objectives("gradient"), "predicted_target_mask"
            )
        if not hasattr(self.workflow, "list_xai_methods"):
            self.view.set_perturbation_method_options(
                self.workflow.list_perturbation_methods(),
                "perturb_occlusion",
            )
        self.view.set_gradcam_dataset_options([], None)
        self.view.set_perturbation_dataset_options([], None)
        if options:
            self.workflow.set_config(options[0]["path"])
            self._sync_model_layer_options()
        self.view.set_rotation_speed_label(self.rotation_speed)
        self.view.set_rotation_running(True)
        self.view.renderer.set_rotation_speed(self.rotation_speed)
        self.view.renderer.show_volumes([], [], [])
        self.view.renderer.start_rotation()
        self._sync_volume_list()
        self.view.workspace.set_workspace_payload(renderable_items=[])
        self.view.set_overlay_status_message("")
        self._sync_transfer_editor()

    def on_model_changed(self, _index: int) -> None:
        path = self.view.selected_model_path()
        if path:
            self.workflow.set_config(path)
            self._sync_model_layer_options()

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
        self.selected_xai_dataset_by_family["gradient"] = dataset_id
        self.selected_xai_dataset_by_family["perturbation"] = dataset_id
        self.view.set_rotation_running(True)

    def _on_store_event(self, event: WorkspaceEvent) -> None:
        if event.name == "dataset_added":
            self._sync_selected_dataset_ids()
            self._sync_dataset_controls()
            self._sync_gradcam_controls()
            self._sync_volume_list()
            self._sync_transfer_editor()
            self._render_current_items(camera_policy="reset_if_first_or_empty")
        elif event.name == "volume_upserted":
            self._sync_selected_dataset_ids()
            self._sync_dataset_controls()
            self._sync_gradcam_controls()
            self._sync_volume_list()
            self._sync_transfer_editor()
            if self._scene_signature() != self._last_scene_signature:
                self._render_current_items(camera_policy="preserve")
            else:
                self._sync_workspace_payload()
        elif event.name in {"dataset_deleted", "volume_deleted"}:
            self._sync_selected_dataset_ids()
            self._sync_dataset_controls()
            self._sync_gradcam_controls()
            self._sync_volume_list()
            self._sync_transfer_editor()
            if self._scene_signature() != self._last_scene_signature:
                self._render_current_items(camera_policy="preserve")
            else:
                self._sync_workspace_payload()
        elif event.name == "volume_order_changed":
            self._sync_volume_list()
            if self._scene_signature() != self._last_scene_signature:
                self._render_current_items(camera_policy="preserve")
            else:
                self._sync_workspace_payload()
        elif event.name == "volume_visibility_changed":
            self._sync_volume_list()
            self._apply_transfer_change(
                str(event.payload.get("volume_id", "")), sync_payload=True
            )
        elif event.name == "volume_renamed":
            self._sync_volume_list()
            self._sync_workspace_payload()
        elif event.name == "transfer_changed":
            self._sync_volume_list()
            self._apply_transfer_change(str(event.payload.get("volume_id", "")))
        elif event.name == "selection_changed":
            self._sync_transfer_editor()
            self._sync_volume_list()
            self._sync_workspace_payload()

    def _on_background_error(self, exc: Exception) -> None:
        if not getattr(exc, "skip_error_store", False):
            self.error_store.save(exc, context="background_task")
        self.view.set_rotation_running(False)

    def _render_current_items(self, *, camera_policy: str = "preserve") -> None:
        self.view.workspace.show_volumes(
            self.data_store.ordered_volume_payloads(),
            self.data_store.ordered_volume_spacing(),
            self.data_store.ordered_volume_metadata(),
            render_settings=self._ordered_render_settings(),
            camera_policy=camera_policy,
        )
        self._last_scene_signature = self._scene_signature()
        self._sync_workspace_payload()

    def _scene_signature(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (
                volume_id,
                tuple(int(v) for v in self.data_store.volumes[volume_id].shape),
            )
            for volume_id in self.data_store.volume_order
            if volume_id in self.data_store.volumes
        )

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
        objective_id: str = "predicted_target_mask",
        method_params: dict[str, object] | None = None,
        on_success=None,
    ) -> None:
        if self._xai_is_running:
            self.view.set_overlay_status_message("XAI is already running.")
            return
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return
        family_id = "perturbation" if method.startswith("perturb") else "gradient"
        self._set_xai_running(True, family_id)

        def _handle_success(result) -> None:
            try:
                (on_success or (lambda value: self._on_xai_result_loaded(dataset_id, value)))(
                    result
                )
            finally:
                self._finish_perturbation_preview()
                self._set_xai_running(False)

        def _handle_error(exc: Exception) -> None:
            try:
                self._on_background_error(exc)
            finally:
                self._finish_perturbation_preview()
                self._set_xai_running(False)

        def _handle_progress(progress: object) -> None:
            if family_id != "perturbation" or not isinstance(progress, dict):
                return
            if progress.get("kind") == "perturb_preview":
                self._show_perturbation_preview(progress)
                return
            current = int(progress.get("current", 0) or 0)
            total = int(progress.get("total", 0) or 0)
            if hasattr(self.view, "set_perturbation_progress"):
                self.view.set_perturbation_progress(current, total)

        def _compute(progress_callback=None):
            active_method_params = dict(method_params or {})
            if family_id == "perturbation" and callable(progress_callback):
                active_method_params["_progress_callback"] = progress_callback
                if self._perturb_preview_enabled:
                    active_method_params["_preview_callback"] = progress_callback
                    active_method_params["_pause_controller"] = (
                        self._perturb_pause_controller
                    )
            return self.workflow.compute_xai(
                dataset.input_state,
                XaiComputeRequest(
                    target_class=target_class,
                    layer=layer,
                    n1=n1,
                    n2=n2,
                    method=method,
                    objective_id=objective_id,
                    result_name=result_name,
                    method_params=active_method_params,
                ),
            )

        try:
            submit = self.task_runner.submit
            if len(signature(submit).parameters) >= 4:
                submit(_compute, _handle_success, _handle_error, _handle_progress)
            else:
                submit(_compute, _handle_success, _handle_error)
        except Exception:
            self._finish_perturbation_preview()
            self._set_xai_running(False)
            raise

    def _set_xai_running(self, is_running: bool, family_id: str = "") -> None:
        self._xai_is_running = bool(is_running)
        self._xai_running_family = str(family_id) if is_running else ""
        for name in ("gradcam_run_button", "perturbation_run_button"):
            button = getattr(self.view, name, None)
            if button is not None and hasattr(button, "setEnabled"):
                button.setEnabled(not self._xai_is_running)
        if hasattr(self.view, "set_perturbation_progress_running"):
            self.view.set_perturbation_progress_running(
                self._xai_is_running and self._xai_running_family == "perturbation"
            )
        if not is_running and hasattr(self.view, "set_perturbation_preview_running"):
            self.view.set_perturbation_preview_running(False)

    def _list_objectives(self, family_id: str | None = None) -> list[dict[str, object]]:
        try:
            return self.workflow.list_objectives(family_id)
        except TypeError:
            return self.workflow.list_objectives()

    def on_layer_changed(self, _layer: str) -> None:
        return

    def on_feature_range_changed(self) -> None:
        return

    def on_class_changed(self, _value: int) -> None:
        return

    def on_method_changed(self, _index: int) -> None:
        self.on_xai_method_changed("gradient")

    def on_xai_method_changed(self, family_id: str) -> None:
        if hasattr(self.view, "selected_xai_method_uses_layer_controls"):
            uses_layer_controls = self.view.selected_xai_method_uses_layer_controls(
                family_id
            )
        elif family_id == "gradient" and hasattr(
            self.view, "selected_method_uses_layer_controls"
        ):
            uses_layer_controls = self.view.selected_method_uses_layer_controls()
        else:
            uses_layer_controls = True
        if family_id == "gradient":
            self.view.set_gradcam_layer_controls_enabled(uses_layer_controls)

    def on_transfer_function_changed(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> None:
        if not self.data_store.update_current_transfer_state(
            transfer_function, data_range
        ):
            return

    def on_transfer_function_change_finished(
        self, _transfer_function: TransferFunction, _data_range: DataRange
    ) -> None:
        self._sync_workspace_payload()

    def _sync_workspace_payload(self) -> None:
        self.view.workspace.set_workspace_payload(
            renderable_items=self.data_store.workspace_renderable_items()
        )
        self.view.set_overlay_status_message(self.view.workspace.overlay_status_message())
        self.refresh_roi_panel()

    def _ordered_render_settings(self) -> list[dict[str, object]]:
        settings = []
        render_items = self.render_items
        for volume_id in self.data_store.volume_order:
            item = render_items.get(volume_id)
            if item is None:
                continue
            transfer_function = item["transfer_function"]
            data_range = item["data_range"]
            colors, opacities = transfer_function.renderer_points(data_range)
            settings.append(
                {
                    "color": colors,
                    "opacity": opacities,
                    "visible": self._volume_visible(volume_id),
                }
            )
        return settings

    def _apply_transfer_change(self, volume_id: str, *, sync_payload: bool = False) -> None:
        try:
            index = self.data_store.volume_order.index(volume_id)
        except ValueError:
            return
        item = self.render_items.get(volume_id)
        if item is None:
            return
        transfer_function = item["transfer_function"]
        data_range = item["data_range"]
        colors, opacities = transfer_function.renderer_points(data_range)
        self.view.workspace.set_volume_transfer_functions(
            index,
            colors,
            opacities,
            visible=self._volume_visible(volume_id),
            render=True,
        )
        if sync_payload:
            self._sync_workspace_payload()

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

    def on_volume_save_requested(self, volume_id: str) -> None:
        try:
            volume = self.data_store.volumes.get(volume_id)
            if volume is None:
                return
            default_name = f"{Path(volume.display_name).stem}.nii.gz"
            if hasattr(self.view, "choose_volume_save_file"):
                path = self.view.choose_volume_save_file(default_name)
            else:
                path = ""
            if not path:
                return
            self.volume_service.save(volume, path)
        except Exception as exc:
            self.error_store.save(exc, context="save_volume")

    def on_gradcam_dataset_changed(self, _index: int) -> None:
        self.selected_grad_dataset_id = self._selected_xai_dataset("gradient")
        self.selected_xai_dataset_by_family["gradient"] = self.selected_grad_dataset_id
        self.data_store.set_active_dataset("gradcam", self.selected_grad_dataset_id)
        self._sync_xai_controls("gradient")

    def on_perturbation_dataset_changed(self, _index: int) -> None:
        self.selected_perturbation_dataset_id = self._selected_xai_dataset(
            "perturbation"
        )
        self.selected_xai_dataset_by_family["perturbation"] = (
            self.selected_perturbation_dataset_id
        )
        self.data_store.set_active_dataset(
            "perturbation", self.selected_perturbation_dataset_id
        )

    def on_gradcam_run_requested(self) -> None:
        self.on_xai_run_requested("gradient")

    def on_xai_run_requested(self, family_id: str) -> None:
        dataset_id = self._selected_xai_dataset(family_id)
        if not dataset_id:
            return
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return
        method = self._selected_xai_method(family_id)
        objective_id = self._selected_xai_objective(family_id)
        model_name = self._selected_model_name()
        uses_layer_controls = self._selected_xai_method_uses_layer_controls(family_id)
        layer = self._selected_xai_layer(family_id) if uses_layer_controls else "input"
        target_class = self._selected_xai_class(family_id)
        result_name = self._prediction_result_name(
            dataset.name,
            model_name,
            layer,
            target_class,
        )
        n1, n2 = (
            self._selected_xai_feature_range(family_id)
            if uses_layer_controls
            else (0, 1)
        )
        if uses_layer_controls and (int(dataset.feature_size or 0) <= 0 or n2 <= n1):
            n1, n2 = 0, 999
        method_params = self._selected_xai_method_params(family_id)
        if family_id == "perturbation":
            self._perturb_preview_enabled = self._selected_perturbation_preview_enabled()
            self._perturb_pause_controller = (
                PerturbationPauseController() if self._perturb_preview_enabled else None
            )
            if self._perturb_preview_enabled:
                self._prepare_camera_for_perturbation_preview()
                self._set_data_controls_enabled(False)
                self._set_camera_controls_enabled(False)
                if hasattr(self.view, "set_perturbation_preview_running"):
                    self.view.set_perturbation_preview_running(True)
            answer_volume_id = self._selected_perturbation_answer_data()
            self.selected_perturbation_answer_volume_id = answer_volume_id
            answer_volume = self.data_store.volumes.get(answer_volume_id)
            if answer_volume is not None:
                method_params["answer_data"] = self._perturbation_answer_data(
                    answer_volume
                )
                method_params["answer_volume_id"] = answer_volume_id
                method_params["answer_volume_name"] = answer_volume.display_name
        method_params.update(
            {
                "model_name": model_name,
                "requested_layer": layer,
                "target_class": target_class,
                "objective_id": objective_id,
            }
        )
        self._run_dataset_method(
            dataset_id,
            target_class=target_class,
            layer=layer,
            n1=n1,
            n2=n2,
            method=method,
            objective_id=objective_id,
            result_name=result_name,
            method_params=method_params,
        )

    def _selected_perturbation_preview_enabled(self) -> bool:
        if hasattr(self.view, "selected_perturbation_preview_enabled"):
            return bool(self.view.selected_perturbation_preview_enabled())
        return False

    def on_perturbation_preview_pause_toggled(self, paused: bool) -> None:
        if self._perturb_pause_controller is not None:
            self._perturb_pause_controller.set_paused(bool(paused))
        if self._perturb_preview_enabled:
            self._set_camera_controls_enabled(bool(paused))

    def _set_data_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self.view, "set_data_controls_enabled"):
            self.view.set_data_controls_enabled(enabled)
            return
        for name in ("volume_list", "delete_volume_button", "save_volume_button", "transfer_editor"):
            control = getattr(self.view, name, None)
            if control is not None and hasattr(control, "setEnabled"):
                control.setEnabled(enabled)

    def _set_camera_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self.view, "set_camera_controls_enabled"):
            self.view.set_camera_controls_enabled(enabled)
            return
        for name in (
            "speed_slider",
            "start_button",
            "stop_button",
            "replace_camera_button",
            "import_camera_button",
            "export_camera_button",
        ):
            control = getattr(self.view, name, None)
            if control is not None and hasattr(control, "setEnabled"):
                control.setEnabled(enabled)

    def _prepare_camera_for_perturbation_preview(self) -> None:
        renderer = getattr(self.view, "renderer", None)
        self._pre_perturb_preview_rotation_running = bool(
            getattr(renderer, "rotating", False)
        )
        stop = getattr(renderer, "stop_rotation", None)
        if callable(stop):
            stop()
        if hasattr(self.view, "set_rotation_running"):
            self.view.set_rotation_running(False)

    def _restore_camera_after_perturbation_preview(self) -> None:
        if self._pre_perturb_preview_rotation_running:
            start = getattr(self.view.renderer, "start_rotation", None)
            if callable(start):
                start()
            if hasattr(self.view, "set_rotation_running"):
                self.view.set_rotation_running(True)
        self._pre_perturb_preview_rotation_running = False

    def _camera_control_allowed(self) -> bool:
        if not self._perturb_preview_enabled:
            return True
        if self._perturb_pause_controller is None:
            return False
        return self._perturb_pause_controller.is_paused()

    def _show_perturbation_preview(self, payload: dict[str, object]) -> None:
        data = payload.get("data")
        if data is None:
            return
        spacing = payload.get("spacing") or (1.0, 1.0, 1.0)
        metadata = dict(payload.get("metadata") or {})
        data_range = DataRange.from_data([data], method="minmax")
        transfer_function = self._perturbation_input_transfer_function()
        colors, opacities = transfer_function.renderer_points(data_range)
        spacing_tuple = tuple(float(v) for v in spacing)
        updated = self._update_perturbation_preview_volume(
            data, spacing_tuple, metadata, colors, opacities
        )
        if not updated:
            self.view.workspace.show_volumes(
                [data],
                [spacing_tuple],
                [metadata],
                render_settings=[
                    {"color": colors, "opacity": opacities, "visible": True}
                ],
                camera_policy="preserve",
            )
        self._perturb_preview_active = True
        rendered_by_box = self._set_preview_box(payload.get("preview_box"))
        if updated and not rendered_by_box:
            renderer = getattr(self.view.workspace, "renderer", None)
            render = getattr(renderer, "render", None)
            if callable(render):
                render()

    def _update_perturbation_preview_volume(
        self,
        data: object,
        spacing: tuple[float, float, float],
        metadata: dict[str, object],
        colors: list[tuple[float, float, float, float]],
        opacities: list[tuple[float, float]],
    ) -> bool:
        if not self._perturb_preview_active:
            return False
        updater = getattr(self.view.workspace, "update_volume_data", None)
        if not callable(updater):
            return False
        if not updater(0, data, spacing, metadata, render=False):
            return False
        self.view.workspace.set_volume_transfer_functions(
            0, colors, opacities, visible=True, render=False
        )
        return True

    def _perturbation_input_transfer_function(self) -> TransferFunction:
        dataset = self.datasets.get(self.selected_perturbation_dataset_id)
        if dataset is None:
            return TransferFunction.base_preset()
        base_volume = self.data_store.volumes.get(dataset.base_volume_id)
        if base_volume is None:
            return TransferFunction.base_preset()
        return base_volume.transfer_function

    def _set_preview_box(self, preview_box: object) -> bool:
        renderer = getattr(self.view.workspace, "renderer", None)
        if renderer is None or not hasattr(renderer, "preview_box"):
            return False
        renderer.preview_box = self._normalized_preview_box(preview_box)
        updater = getattr(renderer, "_update_preview_box", None)
        if callable(updater):
            updater()
            return True
        return False

    @staticmethod
    def _normalized_preview_box(preview_box: object):
        if not isinstance(preview_box, (tuple, list)) or len(preview_box) != 2:
            return None
        try:
            start = tuple(float(v) for v in preview_box[0])
            end = tuple(float(v) for v in preview_box[1])
        except (TypeError, ValueError):
            return None
        if len(start) != 3 or len(end) != 3:
            return None
        return start, end

    def _finish_perturbation_preview(self) -> None:
        if self._perturb_preview_enabled:
            self._set_data_controls_enabled(True)
            self._set_camera_controls_enabled(True)
            self._restore_camera_after_perturbation_preview()
        if self._perturb_preview_active:
            self._set_preview_box(None)
            self._render_current_items(camera_policy="preserve")
        if self._perturb_pause_controller is not None:
            self._perturb_pause_controller.set_paused(False)
        self._perturb_pause_controller = None
        self._perturb_preview_enabled = False
        self._perturb_preview_active = False

    def on_perturbation_run_requested(self) -> None:
        self.on_xai_run_requested("perturbation")

    def _on_xai_result_loaded(self, dataset_id: str, result) -> None:
        result_id = self.data_store.upsert_xai_result(dataset_id, result)
        if result_id is None:
            return
        method_id = (
            result.volume.method_id
            if hasattr(result, "volume")
            else str(result["renderable_item"]["method_id"])
        )
        family_id = "perturbation" if method_id.startswith("perturb") else "gradient"
        if hasattr(self.view, "set_xai_method_options"):
            self.view.set_xai_method_options(
                family_id, list(result.method_options), result.selected_method
            )
            if hasattr(self.view, "set_xai_objective_options"):
                self.view.set_xai_objective_options(
                    family_id, list(result.objective_options), result.selected_objective
                )
            else:
                self.view.set_objective_options(
                    list(result.objective_options), result.selected_objective
                )
            if hasattr(self.view, "set_xai_layer_options"):
                self.view.set_xai_layer_options(
                    family_id,
                    list(result.layer_names),
                    result.selected_layer,
                    result.feature_size,
                )
            elif family_id == "gradient":
                self.view.set_layer_options(list(result.layer_names), result.selected_layer)
                self.view.set_feature_size(result.feature_size)
            self.on_xai_method_changed(family_id)
        elif not method_id.startswith("perturb"):
            self.view.set_method_options(
                list(result.method_options), result.selected_method
            )
            self.view.set_objective_options(
                list(result.objective_options), result.selected_objective
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
        if hasattr(self.view, "set_perturbation_answer_data_options"):
            volume_options = self.data_store.volume_options()
            valid_volume_ids = {option["id"] for option in volume_options}
            if self.selected_perturbation_answer_volume_id not in valid_volume_ids:
                self.selected_perturbation_answer_volume_id = ""
            self.view.set_perturbation_answer_data_options(
                volume_options, self.selected_perturbation_answer_volume_id
            )

    def _sync_gradcam_controls(self) -> None:
        self._sync_xai_controls("gradient")
        self._sync_xai_controls("perturbation")

    def _sync_model_layer_options(self) -> None:
        if not hasattr(self.workflow, "list_current_model_layers"):
            return
        metadata = self.workflow.list_current_model_layers()
        layer_names = list(metadata.get("layer_names", []))
        selected_layer = str(metadata.get("selected_layer", "") or "")
        feature_size = int(metadata.get("feature_size", 0) or 0)
        if hasattr(self.view, "set_xai_layer_options"):
            for family_id in ("gradient", "perturbation"):
                self.view.set_xai_layer_options(
                    family_id, layer_names, selected_layer, feature_size
                )
                self.on_xai_method_changed(family_id)
            return
        self.view.set_layer_options(layer_names, selected_layer)
        self.view.set_feature_size(feature_size)
        self.on_method_changed(0)

    def _sync_xai_controls(self, family_id: str) -> None:
        selected_id = self.selected_xai_dataset_by_family.get(family_id, "")
        if family_id == "gradient":
            selected_id = self.selected_grad_dataset_id
        elif family_id == "perturbation":
            selected_id = self.selected_perturbation_dataset_id
        dataset = self.datasets.get(selected_id)
        if dataset is None:
            if hasattr(self.view, "set_xai_layer_options"):
                self.view.set_xai_layer_options(family_id, [], "", 0)
            elif family_id == "gradient":
                self.view.set_layer_options([], "")
                self.view.set_feature_size(0)
                self.on_method_changed(0)
            return
        if hasattr(self.view, "set_xai_layer_options"):
            self.view.set_xai_layer_options(
                family_id,
                list(dataset.layer_names),
                dataset.selected_layer,
                int(dataset.feature_size),
            )
            self.on_xai_method_changed(family_id)
        elif family_id == "gradient":
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
        self.selected_xai_dataset_by_family["gradient"] = self.selected_grad_dataset_id
        self.selected_xai_dataset_by_family["perturbation"] = (
            self.selected_perturbation_dataset_id
        )

    def _selected_xai_dataset(self, family_id: str) -> str:
        if hasattr(self.view, "selected_xai_dataset"):
            return self.view.selected_xai_dataset(family_id)
        if family_id == "perturbation":
            return self.view.selected_perturbation_dataset()
        return self.view.selected_gradcam_dataset()

    def _selected_xai_method(self, family_id: str) -> str:
        if hasattr(self.view, "selected_xai_method"):
            return self.view.selected_xai_method(family_id)
        if family_id == "perturbation":
            return self.view.selected_perturbation_method()
        return self.view.selected_method()

    def _selected_xai_objective(self, family_id: str) -> str:
        if hasattr(self.view, "selected_xai_objective"):
            return self.view.selected_xai_objective(family_id)
        if family_id == "gradient":
            return self.view.selected_objective()
        return "predicted_mask_dice"

    def _selected_xai_class(self, family_id: str) -> int:
        if hasattr(self.view, "selected_xai_class"):
            return self.view.selected_xai_class(family_id)
        if family_id == "perturbation":
            return int(self.view.perturbation_class_spinbox.value())
        return self.view.selected_class()

    def _selected_xai_layer(self, family_id: str) -> str:
        if hasattr(self.view, "selected_xai_layer"):
            return self.view.selected_xai_layer(family_id)
        if family_id == "perturbation":
            dataset = self.datasets.get(self._selected_xai_dataset(family_id))
            return str(dataset.selected_layer if dataset is not None else "")
        return self.view.selected_layer()

    def _selected_xai_feature_range(self, family_id: str) -> tuple[int, int]:
        if hasattr(self.view, "selected_xai_feature_range"):
            return self.view.selected_xai_feature_range(family_id)
        if family_id == "perturbation":
            dataset = self.datasets.get(self._selected_xai_dataset(family_id))
            return (0, int(dataset.feature_size if dataset is not None else 1))
        return self.view.feature_range()

    def _selected_xai_method_params(self, family_id: str) -> dict[str, object]:
        if hasattr(self.view, "selected_xai_method_params"):
            return dict(self.view.selected_xai_method_params(family_id))
        if family_id == "perturbation":
            return {
                "block_size": self.view.perturbation_block_size_spinbox.value(),
                "stride": self.view.perturbation_stride_spinbox.value(),
            }
        return {}

    def _selected_perturbation_answer_data(self) -> str:
        if hasattr(self.view, "selected_perturbation_answer_data"):
            return self.view.selected_perturbation_answer_data()
        return ""

    def _perturbation_answer_data(self, answer_volume):
        if getattr(answer_volume, "method_id", "") == "base":
            dataset = self.data_store.datasets.get(answer_volume.dataset_id)
            if dataset is not None and dataset.input_state.origin_img is not None:
                return dataset.input_state.origin_img
        return answer_volume.data

    def _selected_xai_method_uses_layer_controls(self, family_id: str) -> bool:
        if hasattr(self.view, "selected_xai_method_uses_layer_controls"):
            return self.view.selected_xai_method_uses_layer_controls(family_id)
        if family_id == "gradient":
            return self.view.selected_method_uses_layer_controls()
        return True

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
        if not self._camera_control_allowed():
            return
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
        if not self._camera_control_allowed():
            return
        self.view.renderer.start_rotation()
        self.view.set_rotation_running(True)

    def on_stop_rotation_requested(self) -> None:
        if not self._camera_control_allowed():
            return
        self.view.renderer.stop_rotation()
        self.view.set_rotation_running(False)

    def on_replace_camera_requested(self) -> None:
        if not self._camera_control_allowed():
            return
        self.view.workspace.replace_camera()

    def on_import_camera_requested(self) -> None:
        if not self._camera_control_allowed():
            return
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
        if not self._camera_control_allowed():
            return
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
