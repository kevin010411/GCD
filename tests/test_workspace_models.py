import unittest

from src.gcd.presentation.qt.workspace_models import (
    SliceOrientation,
    clamp_slice_index,
    collect_slots,
    default_layout_presets,
    orientation_axis,
)


class WorkspaceModelTests(unittest.TestCase):
    def test_layout_presets_cover_expected_workbench_modes(self):
        presets = default_layout_presets()
        preset_ids = {preset.id for preset in presets}
        self.assertEqual(
            preset_ids,
            {"focus_3d", "triple_slice", "quad", "compare"},
        )
        slot_counts = {preset.id: len(collect_slots(preset.root_node)) for preset in presets}
        self.assertEqual(slot_counts["focus_3d"], 3)
        self.assertEqual(slot_counts["triple_slice"], 4)
        self.assertEqual(slot_counts["quad"], 4)
        self.assertEqual(slot_counts["compare"], 3)

    def test_slice_index_is_clamped_to_orientation_dimension(self):
        volume_shape = (128, 96, 72)
        self.assertEqual(
            clamp_slice_index(SliceOrientation.SAGITTAL, 999, volume_shape), 127
        )
        self.assertEqual(
            clamp_slice_index(SliceOrientation.CORONAL, -10, volume_shape), 0
        )
        self.assertEqual(
            clamp_slice_index(SliceOrientation.AXIAL, 80, volume_shape), 80
        )

    def test_orientation_axis_matches_expected_volume_axes(self):
        self.assertEqual(orientation_axis(SliceOrientation.CORONAL), 2)
        self.assertEqual(orientation_axis(SliceOrientation.AXIAL), 1)
        self.assertEqual(orientation_axis(SliceOrientation.SAGITTAL), 0)


if __name__ == "__main__":
    unittest.main()
