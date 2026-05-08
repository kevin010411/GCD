from __future__ import annotations

import unittest

from src.gcd.presentation.qt.annotation_geometry import (
    has_meaningful_3d_box_drag,
    move_rect,
    normalize_rect,
    resize_rect_with_handle,
)


class RoiInteractionGeometryTests(unittest.TestCase):
    def test_normalize_rect_sorts_corners(self) -> None:
        self.assertEqual(normalize_rect((12.0, 20.0, 4.0, 8.0)), (4.0, 8.0, 12.0, 20.0))

    def test_move_rect_preserves_size(self) -> None:
        self.assertEqual(
            move_rect((10.0, 15.0, 30.0, 45.0), (-5.0, 8.0)),
            (5.0, 23.0, 25.0, 53.0),
        )

    def test_resize_rect_with_handle_supports_shrinking_inward(self) -> None:
        self.assertEqual(
            resize_rect_with_handle((10.0, 10.0, 40.0, 50.0), 0, (25.0, 30.0)),
            (25.0, 30.0, 40.0, 50.0),
        )

    def test_resize_rect_with_handle_supports_crossing_over(self) -> None:
        self.assertEqual(
            resize_rect_with_handle((10.0, 10.0, 40.0, 50.0), 0, (60.0, 70.0)),
            (40.0, 50.0, 60.0, 70.0),
        )

    def test_box_creation_requires_non_trivial_drag_distance(self) -> None:
        self.assertFalse(has_meaningful_3d_box_drag((5.0, 5.0, 5.0), (5.2, 5.3, 5.1)))
        self.assertTrue(has_meaningful_3d_box_drag((5.0, 5.0, 5.0), (6.2, 5.3, 5.1)))


if __name__ == "__main__":
    unittest.main()
