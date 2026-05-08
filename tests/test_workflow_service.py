import unittest

import numpy as np

from src.gcd.application.services import WorkflowService
from src.gcd.domain import TransferFunction


class _FakeEngine:
    def __init__(self) -> None:
        self.cam = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        self.volume_data = np.array([-10.0, 10.0, 30.0], dtype=np.float32)
        self.img1_spacing = (1.0, 1.0, 1.0)
        self.display_metadata = {"vtk_origin": (0.0, 0.0, 0.0)}
        self.layers = {"layer-a": 8}

    def compute_cam(self, *, layer, n1, n2):
        return "layer-a"


class WorkflowServiceTests(unittest.TestCase):
    def test_compute_cam_returns_separate_transfer_defaults_and_ranges(self) -> None:
        service = WorkflowService(_FakeEngine())

        result = service.compute_cam(layer=None, n1=0, n2=8)

        self.assertEqual(result["selected_layer"], "layer-a")
        self.assertEqual(result["cam_data_range"].min_value, 0.0)
        self.assertEqual(result["cam_data_range"].max_value, 4.0)
        self.assertEqual(result["volume_data_range"].min_value, -10.0)
        self.assertEqual(result["volume_data_range"].max_value, 30.0)
        self.assertEqual(
            result["cam_transfer_function"].control_points,
            TransferFunction.heatmap_preset().control_points,
        )
        self.assertEqual(
            result["volume_transfer_function"].control_points,
            TransferFunction.base_preset().control_points,
        )


if __name__ == "__main__":
    unittest.main()
