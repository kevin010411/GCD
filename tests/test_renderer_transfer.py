import unittest

from src.gcd.infrastructure.renderer import StandardMultiVolumeRenderer


class _FakeVolume:
    def __init__(self) -> None:
        self.visibility_calls = []
        self.modified_calls = 0

    def SetVisibility(self, value) -> None:
        self.visibility_calls.append(value)

    def Modified(self) -> None:
        self.modified_calls += 1


class _FakeBoundVolume:
    def __init__(self, bounds, visible=True, user_matrix=None) -> None:
        self._bounds = bounds
        self._visible = visible
        self._user_matrix = user_matrix

    def GetBounds(self):
        return self._bounds

    def GetVisibility(self):
        return self._visible

    def GetUserMatrix(self):
        return self._user_matrix


class _FakeImage:
    def __init__(self, bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)) -> None:
        self._bounds = bounds

    def GetBounds(self):
        return self._bounds


class _FakeMatrix:
    def __init__(self, values) -> None:
        self.values = values

    def GetElement(self, row, col):
        return self.values[row][col]


class _FakeProperty:
    def __init__(self) -> None:
        self.modified_calls = 0

    def Modified(self) -> None:
        self.modified_calls += 1


class StandardMultiVolumeTransferTests(unittest.TestCase):
    def _renderer_with_one_volume(self):
        renderer = object.__new__(StandardMultiVolumeRenderer)
        renderer.volumes = [
            {
                "volume": _FakeVolume(),
                "prop": _FakeProperty(),
                "color": [],
                "opacity": [],
                "visible": True,
                "render_port": None,
            }
        ]
        renderer.transfer_apply_calls = []
        renderer.rebuild_visible_calls = 0
        renderer.rebuild_annotation_calls = 0
        renderer.render_calls = 0

        def _apply_transfer(_prop, color, opacity):
            renderer.transfer_apply_calls.append((color, opacity))

        def _rebuild_visible():
            renderer.rebuild_visible_calls += 1

        def _rebuild_annotations():
            renderer.rebuild_annotation_calls += 1

        def _render():
            renderer.render_calls += 1

        renderer._apply_transfer_functions_to_property = _apply_transfer
        renderer._rebuild_visible_multi_volume = _rebuild_visible
        renderer._rebuild_annotation_actors = _rebuild_annotations
        renderer.render = _render
        return renderer

    def test_transfer_only_update_does_not_rebuild_annotations(self) -> None:
        renderer = self._renderer_with_one_volume()

        renderer.set_volume_transfer_functions(
            0,
            [(0.0, 1.0, 0.0, 0.0)],
            [(0.0, 0.5)],
            render=False,
        )

        self.assertEqual(renderer.rebuild_visible_calls, 0)
        self.assertEqual(renderer.rebuild_annotation_calls, 0)
        self.assertEqual(renderer.render_calls, 0)

    def test_unchanged_visibility_does_not_rebuild_visible_backend(self) -> None:
        renderer = self._renderer_with_one_volume()

        renderer.set_volume_transfer_functions(
            0,
            [(0.0, 1.0, 0.0, 0.0)],
            [(0.0, 0.5)],
            visible=True,
            render=False,
        )

        self.assertEqual(renderer.rebuild_visible_calls, 0)
        self.assertEqual(renderer.rebuild_annotation_calls, 0)
        self.assertEqual(renderer.volumes[0]["volume"].visibility_calls, [])

    def test_show_volumes_passes_initial_transfer_and_visibility(self) -> None:
        renderer = object.__new__(StandardMultiVolumeRenderer)
        renderer.add_volume_calls = []
        renderer.clear_calls = []
        renderer.render_calls = 0
        renderer.rebuild_visible_calls = 0

        def _clear(*, render=True):
            renderer.clear_calls.append(render)

        def _add_volume(*args, **kwargs):
            renderer.add_volume_calls.append((args, kwargs))
            return len(renderer.add_volume_calls) - 1

        def _rebuild_visible():
            renderer.rebuild_visible_calls += 1

        def _render():
            renderer.render_calls += 1

        renderer.clear_volumes = _clear
        renderer.add_volume_data = _add_volume
        renderer._rebuild_visible_multi_volume = _rebuild_visible
        renderer.render = _render

        renderer.show_volumes(
            [object()],
            [(1.0, 1.0, 1.0)],
            [None],
            render_settings=[
                {
                    "color": [(0.0, 1.0, 0.0, 0.0)],
                    "opacity": [(0.0, 0.25)],
                    "visible": False,
                }
            ],
        )

        self.assertEqual(renderer.clear_calls, [False])
        self.assertEqual(renderer.render_calls, 1)
        self.assertEqual(renderer.rebuild_visible_calls, 1)
        _args, kwargs = renderer.add_volume_calls[0]
        self.assertEqual(kwargs["color_settings"], [(0.0, 1.0, 0.0, 0.0)])
        self.assertEqual(kwargs["opacity_settings"], [(0.0, 0.25)])
        self.assertFalse(kwargs["visible"])
        self.assertFalse(kwargs["render"])

    def test_camera_reset_uses_visible_volume_flags(self) -> None:
        renderer = object.__new__(StandardMultiVolumeRenderer)
        renderer.volumes = [
            {
                "volume": _FakeBoundVolume(
                    (-100.0, 100.0, -100.0, 100.0, -100.0, 100.0)
                ),
                "image": _FakeImage(),
                "visible": False,
            },
            {
                "volume": _FakeBoundVolume((10.0, 20.0, 30.0, 50.0, 2.0, 6.0)),
                "image": _FakeImage(),
                "visible": True,
            },
        ]

        snapshot = renderer.camera_state_for_visible_volumes()

        self.assertEqual(snapshot["focal_point"], (15.0, 40.0, 4.0))
        self.assertEqual(snapshot["parallel_scale"], 10.0)

    def test_volume_bounds_fallback_applies_user_matrix(self) -> None:
        matrix = _FakeMatrix(
            (
                (0.0, 0.0, -1.0, 40.0),
                (0.0, 1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 20.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        volume = {
            "volume": _FakeBoundVolume(
                (1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
                user_matrix=matrix,
            ),
            "image": _FakeImage((10.0, 16.0, 20.0, 26.0, 30.0, 34.0)),
        }

        bounds = StandardMultiVolumeRenderer._volume_bounds(volume)

        self.assertEqual(bounds, (6.0, 10.0, 20.0, 26.0, 30.0, 36.0))


if __name__ == "__main__":
    unittest.main()
