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
        self.active_method_id = "gradcam"
        self.compute_cam_calls = []
        self.load_input_calls = []

    def available_cam_methods(self):
        return [{"id": "gradcam", "name": "Grad-CAM"}]

    def load_and_process_input(self, file_name, method=None):
        self.load_input_calls.append((file_name, method))
        self.active_method_id = method or "gradcam"
        return ["ok"]

    def set_target_class(self, target_class):
        self.target_class = target_class

    def default_feature_size(self):
        return 8

    def compute_cam(self, *, layer, n1, n2, method=None):
        self.compute_cam_calls.append((layer, n1, n2, method))
        self.active_method_id = method or "gradcam"
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
        self.assertEqual(result["selected_method"], "gradcam")
        self.assertEqual(
            result["method_options"], [{"id": "gradcam", "name": "Grad-CAM"}]
        )
        self.assertEqual(service.engine.compute_cam_calls, [(None, 0, 8, None)])

    def test_load_input_passes_method_through_engine_and_result(self) -> None:
        service = WorkflowService(_FakeEngine())

        result = service.load_input("sample.nii.gz", 3, method="gradcam")

        self.assertEqual(service.engine.load_input_calls, [("sample.nii.gz", "gradcam")])
        self.assertEqual(service.engine.compute_cam_calls, [(None, 0, 8, "gradcam")])
        self.assertEqual(result["selected_method"], "gradcam")
        self.assertEqual(result["messages"], ["ok"])


if __name__ == "__main__":
    unittest.main()
