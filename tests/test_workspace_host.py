import types
import unittest

from src.gcd.presentation.qt.workspace import WorkspaceHost
from src.gcd.presentation.qt.workspace_models import WorkspaceMode


class _FakeRenderer:
    def __init__(self, volume_count=0) -> None:
        self.volumes = [{} for _ in range(volume_count)]
        self.render_calls = 0
        self.applied_snapshots = []

    def show_volumes(self, volumes, spacing, metadata=None) -> None:
        del spacing, metadata
        self.volumes = [{} for _ in volumes]

    def render(self) -> None:
        self.render_calls += 1

    def apply_camera_state(self, snapshot) -> None:
        self.applied_snapshots.append(snapshot)

    def set_volume_transfer_functions(self, *args, **kwargs) -> None:
        del args, kwargs

    def camera_state_for_visible_volumes(self):
        return {
            "position": (1.0, 2.0, 3.0),
            "focal_point": (0.0, 0.0, 0.0),
            "view_up": (0.0, 1.0, 0.0),
            "parallel_scale": 1.0,
        }


class _FakeWorkspace:
    def __init__(self, renderer) -> None:
        self.renderer = renderer
        self.payloads = []
        self.applied_layouts = []
        self.current_preset = types.SimpleNamespace(id="focus_3d", title="Focus 3D")

    def set_workspace_payload(self, **kwargs) -> None:
        self.payloads.append(kwargs)

    def apply_layout(self, preset_id: str) -> None:
        self.applied_layouts.append(preset_id)
        self.current_preset = types.SimpleNamespace(id=preset_id, title=preset_id)

    def apply_slice_snapshot(self, _snapshot) -> None:
        pass

    def capture_slice_snapshot(self):
        return {}


class WorkspaceHostTests(unittest.TestCase):
    def _make_host(
        self,
        *,
        initialized: bool,
        previous_count: int,
        snapshot=None,
        scene_signature=(),
    ):
        host = types.SimpleNamespace()
        host.standard_workspace = _FakeWorkspace(_FakeRenderer(previous_count))
        host.roi_workspace = _FakeWorkspace(_FakeRenderer(previous_count))
        host.shared_state = types.SimpleNamespace(
            camera_snapshot=snapshot,
            slice_snapshot=None,
            renderable_items=[],
        )
        host._scene_initialized = initialized
        host._scene_signature = scene_signature
        host.apply_shared_calls = 0
        host.sync_camera_calls = 0

        def _apply_shared_snapshot_to(_workspace) -> None:
            host.apply_shared_calls += 1

        def sync_camera_to_visible_volumes() -> None:
            host.sync_camera_calls += 1

        host._apply_shared_snapshot_to = _apply_shared_snapshot_to
        host.sync_camera_to_visible_volumes = sync_camera_to_visible_volumes
        return host

    def test_show_volumes_first_load_syncs_camera(self) -> None:
        host = self._make_host(initialized=False, previous_count=0, snapshot={"position": (9, 9, 9)})

        WorkspaceHost.show_volumes(host, [object()], [(1.0, 1.0, 1.0)], [None])

        self.assertEqual(host.sync_camera_calls, 1)
        self.assertEqual(host.apply_shared_calls, 0)
        self.assertTrue(host._scene_initialized)

    def test_show_volumes_reuses_snapshot_when_scene_shape_is_stable(self) -> None:
        host = self._make_host(
            initialized=True,
            previous_count=1,
            snapshot={"position": (9, 9, 9)},
            scene_signature=(("0", ()),),
        )

        WorkspaceHost.show_volumes(host, [object()], [(1.0, 1.0, 1.0)], [None])

        self.assertEqual(host.sync_camera_calls, 0)
        self.assertEqual(host.apply_shared_calls, 2)

    def test_apply_layout_updates_standard_and_roi_workspaces(self) -> None:
        host = self._make_host(initialized=False, previous_count=0)
        host.mode = WorkspaceMode.STANDARD
        host.layout_changed = types.SimpleNamespace(emit=lambda *_args: None)

        WorkspaceHost.apply_layout(host, "quad")

        self.assertEqual(host.standard_workspace.applied_layouts, ["quad"])
        self.assertEqual(host.roi_workspace.applied_layouts, ["quad"])

    def test_switching_to_roi_uses_standard_workspace_layout(self) -> None:
        host = self._make_host(initialized=False, previous_count=0)
        host.mode = WorkspaceMode.STANDARD
        host.active_workspace = host.standard_workspace
        host.stack = types.SimpleNamespace(setCurrentWidget=lambda *_args: None)
        host.standard_workspace.current_preset = types.SimpleNamespace(
            id="compare",
            title="Compare",
        )
        host.standard_workspace.renderer.rotating = False
        host.standard_workspace.renderer.stop_rotation = lambda: None
        host.standard_workspace.renderer.capture_camera_state = lambda: {"position": (1, 2, 3)}
        host.roi_workspace.renderer.rotating = False
        host.roi_workspace.renderer.start_rotation = lambda: None

        WorkspaceHost.set_workspace_mode(host, WorkspaceMode.ROI)

        self.assertEqual(host.roi_workspace.applied_layouts, ["compare"])
        self.assertEqual(host.mode, WorkspaceMode.ROI)


if __name__ == "__main__":
    unittest.main()
