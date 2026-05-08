from __future__ import annotations

import unittest

from src.gcd.application.services import AnnotationJsonService
from src.gcd.presentation.qt.workspace_models import (
    AnnotationMode,
    AnnotationState,
    Box2DAnnotation,
    Box3DAnnotation,
    PointAnnotation,
    SliceOrientation,
)


class AnnotationJsonServiceTests(unittest.TestCase):
    def test_roundtrip_annotation_state(self) -> None:
        service = AnnotationJsonService()
        state = AnnotationState(
            mode=AnnotationMode.BOX,
            point_size=12,
            points=[
                PointAnnotation(
                    id="pt-1",
                    space="voxel",
                    position=(1.0, 2.0, 3.0),
                    size=12,
                    source_viewer_id="slice-1",
                )
            ],
            boxes_2d=[
                Box2DAnnotation(
                    id="box2d-1",
                    orientation=SliceOrientation.AXIAL,
                    slice_index=32,
                    rect=(10.0, 12.0, 20.0, 24.0),
                    source_viewer_id="slice-1",
                )
            ],
            boxes_3d=[
                Box3DAnnotation(
                    id="box3d-1",
                    min_corner=(1.0, 2.0, 3.0),
                    max_corner=(11.0, 12.0, 13.0),
                    is_roi_target=True,
                )
            ],
            selected_annotation_id="box3d-1",
            active_roi_box_id="box3d-1",
        )

        payload = service.serialize(state)
        restored = service.deserialize(payload)

        self.assertEqual(restored.mode, AnnotationMode.BOX)
        self.assertEqual(restored.point_size, 12)
        self.assertEqual(restored.active_roi_box_id, "box3d-1")
        self.assertEqual(restored.selected_annotation_id, "box3d-1")
        self.assertEqual(restored.points[0].position, (1.0, 2.0, 3.0))
        self.assertEqual(restored.boxes_2d[0].orientation, SliceOrientation.AXIAL)
        self.assertEqual(restored.boxes_3d[0].max_corner, (11.0, 12.0, 13.0))


if __name__ == "__main__":
    unittest.main()
