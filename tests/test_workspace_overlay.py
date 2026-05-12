import unittest

import numpy as np

from src.gcd.domain import DataRange, TransferFunction
from src.gcd.presentation.qt.workspace import _resample_item_slice_to_base
from src.gcd.presentation.qt.workspace_models import SliceOrientation


class WorkspaceOverlayTests(unittest.TestCase):
    def test_same_geometry_slice_uses_direct_extraction(self) -> None:
        volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        affine = np.eye(4, dtype=np.float32)
        base_item = {
            "data": volume,
            "metadata": {"affine": affine},
            "display_name": "Base",
        }
        overlay_item = {
            "data": volume * 2,
            "metadata": {"affine": affine.copy()},
            "display_name": "Overlay",
            "transfer_function": TransferFunction.heatmap_preset(),
            "data_range": DataRange(0.0, 1.0),
        }

        overlay_slice, status = _resample_item_slice_to_base(
            base_item, overlay_item, SliceOrientation.AXIAL, 1
        )

        self.assertEqual(status, None)
        np.testing.assert_allclose(overlay_slice, overlay_item["data"][:, 1, :])

    def test_mismatched_geometry_is_resampled_to_base_slice(self) -> None:
        base_volume = np.ones((4, 4, 4), dtype=np.float32)
        overlay_volume = np.ones((2, 2, 2), dtype=np.float32) * 5.0
        base_item = {
            "data": base_volume,
            "metadata": {"affine": np.eye(4, dtype=np.float32)},
            "display_name": "Base",
        }
        overlay_item = {
            "data": overlay_volume,
            "metadata": {"affine": np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32)},
            "display_name": "Overlay",
            "transfer_function": TransferFunction.heatmap_preset(),
            "data_range": DataRange(0.0, 1.0),
        }

        overlay_slice, status = _resample_item_slice_to_base(
            base_item, overlay_item, SliceOrientation.AXIAL, 1
        )

        self.assertEqual(overlay_slice.shape, (4, 4))
        self.assertIn("resampled", status)


if __name__ == "__main__":
    unittest.main()
