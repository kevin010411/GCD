from __future__ import annotations

import threading
from collections.abc import Callable

from ..domain import DataRange, TransferFunction


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


class PerturbationPreviewAdapter:
    def __init__(
        self,
        *,
        view,
        data_store,
        render_current_items: Callable[..., None],
        set_data_controls_enabled: Callable[[bool], None],
        set_camera_controls_enabled: Callable[[bool], None],
        selected_dataset_id: Callable[[], str],
    ) -> None:
        self.view = view
        self.data_store = data_store
        self._render_current_items = render_current_items
        self._set_data_controls_enabled = set_data_controls_enabled
        self._set_camera_controls_enabled = set_camera_controls_enabled
        self._selected_dataset_id = selected_dataset_id
        self.enabled = False
        self.active = False
        self.pause_controller: PerturbationPauseController | None = None
        self._pre_preview_rotation_running = False

    def begin_if_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.pause_controller = (
            PerturbationPauseController() if self.enabled else None
        )
        if not self.enabled:
            return
        self._prepare_camera()
        self._set_data_controls_enabled(False)
        self._set_camera_controls_enabled(False)
        if hasattr(self.view, "set_perturbation_preview_running"):
            self.view.set_perturbation_preview_running(True)

    def runtime_method_params(self, progress_callback) -> dict[str, object]:
        if not self.enabled or not callable(progress_callback):
            return {}
        return {
            "_preview_callback": progress_callback,
            "_pause_controller": self.pause_controller,
        }

    def pause_toggled(self, paused: bool) -> None:
        if self.pause_controller is not None:
            self.pause_controller.set_paused(bool(paused))
        if self.enabled:
            self._set_camera_controls_enabled(bool(paused))

    def camera_control_allowed(self) -> bool:
        if not self.enabled:
            return True
        if self.pause_controller is None:
            return False
        return self.pause_controller.is_paused()

    def handle_payload(self, payload: dict[str, object]) -> None:
        data = payload.get("data")
        if data is None:
            return
        spacing = payload.get("spacing") or (1.0, 1.0, 1.0)
        metadata = dict(payload.get("metadata") or {})
        data_range = DataRange.from_data([data], method="minmax")
        transfer_function = self._input_transfer_function()
        colors, opacities = transfer_function.renderer_points(data_range)
        spacing_tuple = tuple(float(v) for v in spacing)
        updated = self._update_preview_volume(
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
        self.active = True
        rendered_by_box = self._set_preview_box(payload.get("preview_box"))
        if updated and not rendered_by_box:
            renderer = getattr(self.view.workspace, "renderer", None)
            render = getattr(renderer, "render", None)
            if callable(render):
                render()

    def finish(self) -> None:
        if self.enabled:
            self._set_data_controls_enabled(True)
            self._set_camera_controls_enabled(True)
            self._restore_camera()
        if self.active:
            self._set_preview_box(None)
            self._render_current_items(camera_policy="preserve")
        if self.pause_controller is not None:
            self.pause_controller.set_paused(False)
        self.pause_controller = None
        self.enabled = False
        self.active = False

    def _prepare_camera(self) -> None:
        renderer = getattr(self.view, "renderer", None)
        self._pre_preview_rotation_running = bool(getattr(renderer, "rotating", False))
        stop = getattr(renderer, "stop_rotation", None)
        if callable(stop):
            stop()
        if hasattr(self.view, "set_rotation_running"):
            self.view.set_rotation_running(False)

    def _restore_camera(self) -> None:
        if self._pre_preview_rotation_running:
            start = getattr(self.view.renderer, "start_rotation", None)
            if callable(start):
                start()
            if hasattr(self.view, "set_rotation_running"):
                self.view.set_rotation_running(True)
        self._pre_preview_rotation_running = False

    def _update_preview_volume(
        self,
        data: object,
        spacing: tuple[float, float, float],
        metadata: dict[str, object],
        colors: list[tuple[float, float, float, float]],
        opacities: list[tuple[float, float]],
    ) -> bool:
        if not self.active:
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

    def _input_transfer_function(self) -> TransferFunction:
        dataset = self.data_store.datasets.get(self._selected_dataset_id())
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
        renderer.preview_box = self.normalized_preview_box(preview_box)
        updater = getattr(renderer, "_update_preview_box", None)
        if callable(updater):
            updater()
            return True
        return False

    @staticmethod
    def normalized_preview_box(preview_box: object):
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
