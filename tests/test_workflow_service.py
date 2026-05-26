import unittest

import numpy as np

from src.gcd.application.services import WorkflowService
from src.gcd.domain import DatasetInput, TransferFunction


class _FakeEngine:
    def __init__(self) -> None:
        self.cam = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        self.volume_data = np.array([-10.0, 10.0, 30.0], dtype=np.float32)
        self.img1_spacing = (1.0, 1.0, 1.0)
        self.display_metadata = {
            "vtk_origin": (0.0, 0.0, 0.0),
            "affine": np.eye(4, dtype=np.float32),
        }
        self.layers = {"layer-a": 8}
        self.cfg = {"default_layer": "layer-a"}
        self.active_method_id = "gradcam"
        self.compute_cam_calls = []
        self.load_volume_calls = []
        self.prepare_calls = []
        self.target_class = 0
        self.patch = []
        self.xai_cache_key = ""
        self.prepared_layers = {"layer-a": 8, "layer-b": 4}

    def available_cam_methods(self, category=None):
        if category == "perturbation":
            return [
                {
                    "id": "perturb_occlusion",
                    "name": "Occlusion",
                    "uses_layer_controls": True,
                }
            ]
        return [
            {"id": "gradcam", "name": "Grad-CAM", "uses_layer_controls": True},
            {
                "id": "saliency_map",
                "name": "Saliency Map",
                "uses_layer_controls": False,
            },
        ]

    def load_volume(self, file_name):
        self.load_volume_calls.append(file_name)
        return ["ok"]

    def prepare_xai_inputs(self, method=None):
        self.prepare_calls.append(method)
        self.active_method_id = method or "gradcam"
        self.patch = [{"method": self.active_method_id, "pred": np.zeros((1, 1, 1), dtype=np.float32)}]
        self.layers = {"input": 1} if self.active_method_id == "saliency_map" else dict(self.prepared_layers)
        self.xai_cache_key = f"cfg.py|{self.target_class}|{self.active_method_id}"

    def set_target_class(self, target_class):
        self.target_class = target_class

    def default_feature_size(self):
        return 8

    def dataset_input(self):
        return DatasetInput(
            img0=None,
            img1=np.array([1.0], dtype=np.float32),
            origin_img=None,
            origin_meta={},
            origin_shape=(1,),
            img1_spacing=self.img1_spacing,
            display_metadata=dict(self.display_metadata),
            layers=dict(self.layers),
            file_name="sample.nii.gz",
            target_class=self.target_class,
            active_method_id=self.active_method_id,
            xai_cache_key=self.xai_cache_key,
        )

    def load_dataset_input(self, dataset_input):
        self.target_class = dataset_input.target_class
        self.active_method_id = dataset_input.active_method_id
        self.xai_cache_key = dataset_input.xai_cache_key
        self.layers = dict(dataset_input.layers)
        self.patch = []

    def compute_cam(self, *, layer, n1, n2, method=None, method_params=None):
        self.compute_cam_calls.append((layer, n1, n2, method, method_params))
        self.active_method_id = method or "gradcam"
        if self.active_method_id == "saliency_map":
            self.layers = {"input": 1}
            return "input"
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
            result["method_options"],
            [
                {"id": "gradcam", "name": "Grad-CAM", "uses_layer_controls": True},
                {
                    "id": "saliency_map",
                    "name": "Saliency Map",
                    "uses_layer_controls": False,
                },
            ],
        )
        self.assertEqual(service.engine.compute_cam_calls, [(None, 0, 8, None, None)])

    def test_load_input_passes_method_through_engine_without_auto_compute(self) -> None:
        service = WorkflowService(_FakeEngine())

        result = service.load_input("sample.nii.gz", 3, method="gradcam")

        self.assertEqual(service.engine.load_volume_calls, ["sample.nii.gz"])
        self.assertEqual(service.engine.compute_cam_calls, [])
        self.assertEqual(service.engine.prepare_calls, [])
        self.assertEqual(result["selected_method"], "gradcam")
        self.assertEqual(result["messages"], ["ok"])
        self.assertIs(result["volume_data"], service.engine.volume_data)
        self.assertEqual(result["spacing"], (1.0, 1.0, 1.0))
        self.assertEqual(result["display_metadata"]["vtk_origin"], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(
            result["display_metadata"]["affine"], np.eye(4, dtype=np.float32)
        )

    def test_compute_dataset_result_returns_renderable_item_metadata(self) -> None:
        service = WorkflowService(_FakeEngine())

        result = service.compute_dataset_result(
            service.engine.dataset_input(),
            target_class=1,
            layer="layer-a",
            n1=0,
            n2=8,
            method="perturb_occlusion",
            result_name="sample_model_perturb方法",
            method_params={"block_size": 16},
        )

        self.assertEqual(
            service.engine.compute_cam_calls,
            [("layer-a", 0, 8, "perturb_occlusion", {"block_size": 16})],
        )
        self.assertEqual(service.engine.prepare_calls, ["perturb_occlusion"])
        self.assertEqual(result["renderable_item"]["name"], "sample_model_perturb方法")
        self.assertEqual(result["renderable_item"]["source"], "xai")
        self.assertEqual(result["selected_method"], "perturb_occlusion")

    def test_compute_dataset_result_prepares_when_only_placeholder_layers_exist(self) -> None:
        service = WorkflowService(_FakeEngine())

        service.compute_dataset_result(
            DatasetInput(
                img0=None,
                img1=np.array([1.0], dtype=np.float32),
                origin_img=None,
                origin_meta={},
                origin_shape=(1,),
                img1_spacing=(1.0, 1.0, 1.0),
                display_metadata={},
                layers={"layer-a": 1},
                file_name="sample.nii.gz",
                target_class=1,
                active_method_id="gradcam",
                xai_cache_key="cfg.py|1|gradcam",
            ),
            target_class=1,
            layer="layer-a",
            n1=0,
            n2=8,
            method="gradcam",
            result_name="sample_model_grad方法",
        )

        self.assertEqual(service.engine.prepare_calls, ["gradcam"])

    def test_compute_dataset_result_prepares_when_requested_layer_is_missing(self) -> None:
        service = WorkflowService(_FakeEngine())

        service.compute_dataset_result(
            DatasetInput(
                img0=None,
                img1=np.array([1.0], dtype=np.float32),
                origin_img=None,
                origin_meta={},
                origin_shape=(1,),
                img1_spacing=(1.0, 1.0, 1.0),
                display_metadata={},
                layers={"layer-a": 8},
                file_name="sample.nii.gz",
                target_class=1,
                active_method_id="gradcam",
                xai_cache_key="cfg.py|1|gradcam",
            ),
            target_class=1,
            layer="layer-z",
            n1=0,
            n2=8,
            method="gradcam",
            result_name="sample_model_grad方法",
        )

        self.assertEqual(service.engine.prepare_calls, ["gradcam"])

    def test_compute_dataset_result_uses_input_metadata_for_saliency_map(self) -> None:
        service = WorkflowService(_FakeEngine())

        result = service.compute_dataset_result(
            service.engine.dataset_input(),
            target_class=1,
            layer="layer-a",
            n1=0,
            n2=8,
            method="saliency_map",
            result_name="sample_model_saliency",
        )

        self.assertEqual(service.engine.prepare_calls, ["saliency_map"])
        self.assertEqual(
            service.engine.compute_cam_calls,
            [("layer-a", 0, 8, "saliency_map", None)],
        )
        self.assertEqual(result["layer_names"], ["input"])
        self.assertEqual(result["selected_layer"], "input")
        self.assertEqual(result["feature_size"], 1)
        self.assertEqual(result["selected_method"], "saliency_map")


if __name__ == "__main__":
    unittest.main()
