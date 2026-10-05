import unittest

import numpy as np

from src.gcd.application.services import WorkflowService
from src.gcd.domain import DatasetInput, TransferFunction, XaiComputeRequest


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
        self.active_objective_id = "predicted_target_mask"
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
                    "uses_layer_controls": False,
                    "uses_objective": True,
                    "parameters": [
                        {"id": "block_size", "label": "Block Size", "kind": "int"},
                    ],
                },
                {
                    "id": "perturb_lime",
                    "name": "LIME",
                    "uses_layer_controls": False,
                    "uses_objective": True,
                    "parameters": [
                        {"id": "num_samples", "label": "Samples", "kind": "int"},
                    ],
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

    def available_objectives(self, family=None):
        if family == "perturbation":
            return [
                {"id": "predicted_mask_dice", "name": "Predicted Mask Dice"},
                {"id": "predicted_mask_iou", "name": "Predicted Mask IoU"},
            ]
        return [
            {"id": "predicted_target_mask", "name": "Predicted Target Mask"},
            {"id": "target_logit_sum", "name": "Target Logit Sum"},
        ]

    def load_volume(self, file_name):
        self.load_volume_calls.append(file_name)
        return ["ok"]

    def prepare_xai_inputs(self, method=None, objective_id=None, method_params=None):
        self.prepare_calls.append((method, objective_id, method_params))
        self.active_method_id = method or "gradcam"
        self.active_objective_id = objective_id or "predicted_target_mask"
        self.patch = [{"method": self.active_method_id, "pred": np.zeros((1, 1, 1), dtype=np.float32)}]
        self.layers = (
            {"input": 1}
            if self.active_method_id == "saliency_map"
            or self.active_method_id.startswith("perturb")
            else dict(self.prepared_layers)
        )
        self.xai_cache_key = (
            f"cfg.py|{self.target_class}|{self.active_method_id}|"
            f"{self.active_objective_id}"
        )

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
            active_objective_id=self.active_objective_id,
            xai_cache_key=self.xai_cache_key,
        )

    def load_dataset_input(self, dataset_input):
        self.target_class = dataset_input.target_class
        self.active_method_id = dataset_input.active_method_id
        self.active_objective_id = dataset_input.active_objective_id
        self.xai_cache_key = dataset_input.xai_cache_key
        self.layers = dict(dataset_input.layers)
        self.patch = []

    def compute_cam(self, *, layer, n1, n2, method=None, method_params=None):
        self.compute_cam_calls.append((layer, n1, n2, method, method_params))
        self.active_method_id = method or "gradcam"
        if self.active_method_id == "saliency_map":
            self.layers = {"input": 1}
            return "input"
        if self.active_method_id.startswith("perturb"):
            self.layers = {"input": 1}
            return "input"
        return "layer-a"


class _FakeRunnerEngine(_FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.run_xai_method_calls = []

    def run_xai_method(self, *, layer, n1, n2, method=None, method_params=None):
        self.run_xai_method_calls.append((layer, n1, n2, method, method_params))
        self.active_method_id = method or "gradcam"
        if self.active_method_id == "saliency_map" or self.active_method_id.startswith(
            "perturb"
        ):
            self.layers = {"input": 1}
            return "input"
        return "layer-a"


class WorkflowServiceTests(unittest.TestCase):
    def test_result_keeps_all_layer_choices_and_requested_class(self) -> None:
        class _SelectedLayerEngine(_FakeEngine):
            def prepare_xai_inputs(self, method=None, objective_id=None, method_params=None):
                super().prepare_xai_inputs(method, objective_id, method_params)
                self.layers = {"layer-b": 4}
                self.available_layer_names = ("layer-a", "layer-b")

            def compute_cam(self, *, layer, n1, n2, method=None, method_params=None):
                self.compute_cam_calls.append((layer, n1, n2, method, method_params))
                return "layer-b"

        engine = _SelectedLayerEngine()
        result = WorkflowService(engine).compute_xai(
            engine.dataset_input(),
            XaiComputeRequest(
                target_class=2,
                layer="layer-b",
                n1=0,
                n2=4,
                method="scorecam",
                result_name="sample_scorecam_class2",
            ),
        )

        self.assertEqual(result.layer_names, ("layer-a", "layer-b"))
        self.assertEqual(result.selected_layer, "layer-b")
        self.assertEqual(result.feature_size, 4)
        self.assertEqual(engine.target_class, 2)
        self.assertEqual(result.volume.plugin_metadata["target_class"], 2)

    def test_gradient_result_includes_prediction_volume_for_data_plugin(self) -> None:
        engine = _FakeEngine()
        engine.model_output = np.array([[[0, 1], [2, 1]]], dtype=np.int16)
        service = WorkflowService(engine)

        result = service.compute_xai(
            engine.dataset_input(),
            XaiComputeRequest(
                target_class=1,
                layer="layer-a",
                n1=0,
                n2=8,
                method="gradcam",
                result_name="sample_model_gradcam",
            ),
        )

        self.assertIsNotNone(result.prediction_volume)
        self.assertEqual(result.prediction_volume.source, "prediction")
        self.assertEqual(result.prediction_volume.method_id, "model_prediction")
        self.assertEqual(
            result.prediction_volume.display_name,
            "sample_model_prediction",
        )
        np.testing.assert_array_equal(
            result.prediction_volume.data,
            np.array([[[0, 1], [2, 1]]], dtype=np.int16),
        )

    def test_prediction_name_and_identity_do_not_depend_on_xai_request(self) -> None:
        engine = _FakeEngine()
        engine.model_output = np.array([[[0, 1], [2, 1]]], dtype=np.int16)
        engine.model_identity = lambda: "architecture-and-checkpoint"
        service = WorkflowService(engine)
        for method, target_class, layer in (
            ("gradcam", 1, "layer-a"),
            ("saliency_map", 2, "input"),
            ("perturb_occlusion", 3, "input"),
        ):
            result = service.compute_xai(
                engine.dataset_input(),
                XaiComputeRequest(target_class, layer, 0, 8, method, f"heatmap_{method}_{target_class}", method_params={"model_name": "unet"}),
            )
            self.assertEqual(result.prediction_volume.display_name, "sample_unet_prediction")
            self.assertEqual(result.prediction_volume.plugin_metadata["model_key"], "architecture-and-checkpoint")
            self.assertEqual(result.prediction_volume.method_id, "model_prediction")

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
        self.assertEqual(result["selected_objective"], "predicted_target_mask")
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
        self.assertEqual(result["layer_names"], [])
        self.assertEqual(result["selected_layer"], "")
        self.assertEqual(result["feature_size"], 0)
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
        self.assertEqual(
            service.engine.prepare_calls,
            [("perturb_occlusion", "predicted_mask_dice", {"block_size": 16})],
        )
        self.assertEqual(result["renderable_item"]["name"], "sample_model_perturb方法")
        self.assertEqual(result["renderable_item"]["source"], "xai")
        self.assertEqual(result["selected_method"], "perturb_occlusion")

    def test_compute_dataset_result_prefers_explicit_xai_runner_entrypoint(self) -> None:
        service = WorkflowService(_FakeRunnerEngine())

        service.compute_dataset_result(
            service.engine.dataset_input(),
            target_class=1,
            layer="layer-a",
            n1=0,
            n2=8,
            method="gradcam",
            result_name="sample_model_grad方法",
        )

        self.assertEqual(
            service.engine.run_xai_method_calls,
            [("layer-a", 0, 8, "gradcam", None)],
        )
        self.assertEqual(service.engine.compute_cam_calls, [])

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

        self.assertEqual(
            service.engine.prepare_calls, [("gradcam", "predicted_target_mask", None)]
        )

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

        self.assertEqual(
            service.engine.prepare_calls, [("gradcam", "predicted_target_mask", None)]
        )

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

        self.assertEqual(
            service.engine.prepare_calls,
            [("saliency_map", "predicted_target_mask", None)],
        )
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
