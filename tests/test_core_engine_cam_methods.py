import unittest
from unittest.mock import patch

import torch

from src.gcd.infrastructure.xai.methods.cam_methods import (
    GradCamMethod,
    PerturbationOcclusionMethod,
    SaliencyMapMethod,
    XResCamMethod,
)
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine


class CoreEngineCamMethodTests(unittest.TestCase):
    def test_model_layer_metadata_reads_nested_targets_without_forward(self) -> None:
        class _Cfg:
            model = object()

            def get(self, key, default=None):
                return {"default_layer": "decoder"}.get(key, default)

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Sequential(torch.nn.Conv3d(1, 2, kernel_size=1))
                self.decoder = torch.nn.Sequential(
                    torch.nn.Identity(),
                    torch.nn.Conv3d(2, 3, kernel_size=1),
                )
                self.xai_layer_targets = {
                    "encoder": "encoder.0",
                    "decoder": "decoder.1",
                }

            def forward(self, _value):
                raise AssertionError("metadata lookup must not run forward")

        engine = GradCamEngine.__new__(GradCamEngine)
        engine.cfg = _Cfg()

        with patch("src.gcd.infrastructure.xai.engine.core_engine._build_model", return_value=_Model()):
            metadata = engine.model_layer_metadata()

        self.assertEqual(metadata["layer_names"], ["encoder", "decoder"])
        self.assertEqual(metadata["selected_layer"], "decoder")
        self.assertEqual(metadata["feature_size"], 0)

    def test_model_layer_metadata_falls_back_to_first_layer(self) -> None:
        class _Cfg:
            model = object()

            def get(self, key, default=None):
                return {"default_layer": "missing"}.get(key, default)

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Conv3d(1, 2, kernel_size=1)
                self.xai_layer_targets = {"encoder": "encoder"}

        engine = GradCamEngine.__new__(GradCamEngine)
        engine.cfg = _Cfg()

        with patch("src.gcd.infrastructure.xai.engine.core_engine._build_model", return_value=_Model()):
            metadata = engine.model_layer_metadata()

        self.assertEqual(metadata["selected_layer"], "encoder")

    def test_model_layer_metadata_reports_missing_target_path(self) -> None:
        class _Cfg:
            model = object()

            def get(self, key, default=None):
                return {"default_layer": "decoder"}.get(key, default)

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Conv3d(1, 2, kernel_size=1)
                self.xai_layer_targets = {"decoder": "decoder.0"}

        engine = GradCamEngine.__new__(GradCamEngine)
        engine.cfg = _Cfg()

        with (
            patch("src.gcd.infrastructure.xai.engine.core_engine._build_model", return_value=_Model()),
            self.assertRaisesRegex(RuntimeError, "decoder.*decoder\\.0"),
        ):
            engine.model_layer_metadata()

    def test_unknown_method_falls_back_to_gradcam(self) -> None:
        logs = []
        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = logs.append
        engine.cam_methods = {"gradcam": GradCamMethod(GradCamEngine._gradcam_objective)}
        engine.active_method_id = "gradcam"

        method = engine._resolve_cam_method("mystery")

        self.assertEqual(method.id, "gradcam")
        self.assertTrue(any("未知 CAM method" in message for message in logs))

    def test_compute_cam_keeps_gradcam_output_normalized(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.cam_methods = {"gradcam": GradCamMethod(GradCamEngine._gradcam_objective)}
        engine.active_method_id = "gradcam"
        engine.file_name = "sample.nii.gz"
        engine.cfg = {"default_layer": "layer-a"}
        engine.layers = {"layer-a": 1}
        engine.SIZE = 2
        engine.STRIDE = 0
        engine.PERMUTE = (0, 1, 2)
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.save_dir = None
        pred = torch.tensor([[[[[0.1]]], [[[0.9]]]]], dtype=torch.float32)
        tile_payload = {
            "method": "gradcam",
            "pred": pred,
            "layers": {
                "layer-a": {
                    "activation": torch.ones((1, 1, 1, 1, 1), dtype=torch.float32),
                    "gradient": torch.full(
                        (1, 1, 1, 1, 1), 2.0, dtype=torch.float32
                    ),
                }
            },
        }
        engine.patch = [tile_payload, tile_payload, tile_payload, tile_payload]

        selected = engine.compute_cam(method="gradcam")

        self.assertEqual(selected, "layer-a")
        self.assertGreaterEqual(float(engine.cam.min()), 0.0)
        self.assertLessEqual(float(engine.cam.max()), 1.0)

    def test_compute_cam_keeps_perturbation_negative_score_changes_visible(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.cam_methods = {"perturb_occlusion": PerturbationOcclusionMethod()}
        engine.active_method_id = "perturb_occlusion"
        engine.file_name = "sample.nii.gz"
        engine.cfg = {"default_layer": "layer-a"}
        engine.layers = {"input": 1}
        engine.SIZE = 2
        engine.STRIDE = 2
        engine.PERMUTE = (0, 1, 2)
        engine.img1 = torch.ones((1, 4, 4, 2), dtype=torch.float32)
        engine.save_dir = None
        pred = torch.tensor([[[[[0.1]]], [[[0.9]]]]], dtype=torch.float32)
        importance = -torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
        importance[:, :, 0, 0, 0] = -2.0
        tile_payload = {
            "method": "perturb_occlusion",
            "pred": pred,
            "importance_map": importance,
        }
        engine.patch = [tile_payload, tile_payload, tile_payload, tile_payload]

        selected = engine.compute_cam(method="perturb_occlusion")

        self.assertEqual(selected, "input")
        self.assertGreater(float(engine.cam.max()), 0.0)
        self.assertGreaterEqual(float(engine.cam.min()), 0.0)
        self.assertLessEqual(float(engine.cam.max()), 1.0)

    def test_compute_cam_shows_uniform_perturbation_signal(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.cam_methods = {"perturb_occlusion": PerturbationOcclusionMethod()}
        engine.active_method_id = "perturb_occlusion"
        engine.file_name = "sample.nii.gz"
        engine.cfg = {"default_layer": "layer-a"}
        engine.layers = {"input": 1}
        engine.SIZE = 2
        engine.STRIDE = 2
        engine.PERMUTE = (0, 1, 2)
        engine.img1 = torch.ones((1, 4, 4, 2), dtype=torch.float32)
        engine.save_dir = None
        pred = torch.tensor([[[[[0.1]]], [[[0.9]]]]], dtype=torch.float32)
        tile_payload = {
            "method": "perturb_occlusion",
            "pred": pred,
            "importance_map": -torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
        }
        engine.patch = [tile_payload, tile_payload, tile_payload, tile_payload]

        engine.compute_cam(method="perturb_occlusion")

        self.assertTrue(torch.allclose(engine.cam, torch.ones_like(engine.cam)))

    def test_available_cam_methods_includes_layer_control_metadata(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine.cam_methods = {
            "gradcam": GradCamMethod(GradCamEngine._gradcam_objective),
            "xrescam": XResCamMethod(GradCamEngine._gradcam_objective),
            "saliency_map": SaliencyMapMethod(GradCamEngine._gradcam_objective),
        }

        methods = engine.available_cam_methods("grad")

        by_id = {method["id"]: method for method in methods}
        self.assertTrue(by_id["gradcam"]["uses_layer_controls"])
        self.assertTrue(by_id["xrescam"]["uses_layer_controls"])
        self.assertEqual(by_id["xrescam"]["name"], "XResCAM")
        self.assertFalse(by_id["saliency_map"]["uses_layer_controls"])

    def test_available_objectives_exposes_gradient_aggregation_choices(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)

        objectives = engine.available_objectives("gradient")

        by_id = {objective["id"]: objective["name"] for objective in objectives}
        self.assertEqual(by_id["predicted_target_mask"], "Predicted Target Mask")
        self.assertEqual(by_id["target_logit_sum"], "Target Logit Sum")
        self.assertEqual(by_id["target_probability_sum"], "Target Probability Sum")
        self.assertEqual(by_id["target_margin"], "Target Margin")

    def test_available_objectives_exposes_perturbation_score_choices(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)

        objectives = engine.available_objectives("perturbation")

        by_id = {objective["id"]: objective["name"] for objective in objectives}
        self.assertEqual(by_id["predicted_mask_dice"], "Predicted Mask Dice")
        self.assertEqual(by_id["predicted_mask_iou"], "Predicted Mask IoU")
        self.assertEqual(by_id["target_probability_sum"], "Target Probability Sum")

    def test_perturbation_runner_pause_waits_only_for_perturbation_methods(self) -> None:
        class _PauseController:
            def __init__(self) -> None:
                self.waits = 0

            def wait_if_paused(self) -> None:
                self.waits += 1

        controller = _PauseController()
        GradCamEngine._wait_for_perturbation_pause(
            type("Method", (), {"family": "perturbation"})(),
            {"_pause_controller": controller},
        )
        GradCamEngine._wait_for_perturbation_pause(
            type("Method", (), {"family": "gradient"})(),
            {"_pause_controller": controller},
        )

        self.assertEqual(controller.waits, 1)

    def test_predicted_mask_scores_compare_prediction_to_reference_mask(self) -> None:
        logits = torch.tensor(
            [[[[[0.1, 3.0]]], [[[2.0, 1.0]]]]],
            dtype=torch.float32,
        )
        reference_mask = torch.tensor([[[True, True]]])

        dice = GradCamEngine._predicted_mask_dice_score(logits, 1, reference_mask)
        iou = GradCamEngine._predicted_mask_iou_score(logits, 1, reference_mask)

        self.assertAlmostEqual(float(dice), 2.0 / 3.0)
        self.assertAlmostEqual(float(iou), 0.5)

    def test_perturb_reference_volume_handles_missing_display_volume_data(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine.img1 = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
        engine.volume_data = None
        engine.PERMUTE = (2, 1, 0)
        engine.target_class = 1
        answer_data = torch.zeros((4, 3, 2), dtype=torch.float32)
        answer_data[1, 1, 0] = 1.0

        reference = engine._perturb_reference_volume(
            {"answer_data": answer_data}, torch.device("cpu")
        )

        self.assertIsNotNone(reference)
        self.assertEqual(tuple(reference.shape), (2, 3, 4))
        self.assertTrue(bool(reference[0, 1, 1]))

    def test_perturb_reference_volume_uses_multiclass_label_map_even_when_target_absent(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine.img1 = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        engine.volume_data = None
        engine.PERMUTE = (0, 1, 2)
        engine.target_class = 3
        answer_data = torch.tensor(
            [[[0.0, 1.0], [2.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]]],
            dtype=torch.float32,
        )

        reference = engine._perturb_reference_volume(
            {"answer_data": answer_data}, torch.device("cpu")
        )

        self.assertIsNotNone(reference)
        self.assertEqual(int(reference.sum()), 0)

    def test_target_logit_sum_objective_aggregates_all_target_voxels(self) -> None:
        logits = torch.tensor(
            [[[[[1.0, 2.0]]], [[[3.0, 4.0]]], [[[5.0, 1.0]]]]],
            dtype=torch.float32,
        )

        loss = GradCamEngine._target_logit_sum_objective(logits, 1)

        self.assertEqual(float(loss), 7.0)

    def test_predicted_target_mask_objective_uses_argmax_mask(self) -> None:
        logits = torch.tensor(
            [[[[[5.0, 2.0]]], [[[3.0, 4.0]]], [[[1.0, 3.0]]]]],
            dtype=torch.float32,
        )

        loss = GradCamEngine._predicted_target_mask_objective(logits, 1)

        self.assertEqual(float(loss), 4.0)

    def test_compute_cam_uses_input_layer_for_saliency_map(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.cam_methods = {
            "saliency_map": SaliencyMapMethod(GradCamEngine._gradcam_objective)
        }
        engine.active_method_id = "saliency_map"
        engine.file_name = "sample.nii.gz"
        engine.cfg = {"default_layer": "layer-a"}
        engine.layers = {"input": 1}
        engine.SIZE = 2
        engine.STRIDE = 0
        engine.PERMUTE = (0, 1, 2)
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.save_dir = None
        pred = torch.tensor([[[[[0.1]]], [[[0.9]]]]], dtype=torch.float32)
        tile_payload = {
            "method": "saliency_map",
            "pred": pred,
            "input_gradient": torch.full((1, 1, 1, 1, 1), 2.0, dtype=torch.float32),
        }
        engine.patch = [tile_payload, tile_payload, tile_payload, tile_payload]

        selected = engine.compute_cam(method="saliency_map")

        self.assertEqual(selected, "input")
        self.assertEqual(engine.layers, {"input": 1})
        self.assertGreaterEqual(float(engine.cam.min()), 0.0)
        self.assertLessEqual(float(engine.cam.max()), 1.0)

    def test_prepare_xai_inputs_allows_input_based_method_without_layers(self) -> None:
        class _Cfg:
            model = object()
            ckpt = "checkpoint.pt"

        class _Model(torch.nn.Module):
            def load_state_dict(self, _state_dict, strict=False):
                return [], []

            def forward(self, value):
                mean = value.mean(dim=(2, 3, 4))
                total = value.sum(dim=(2, 3, 4))
                return torch.cat([mean, total], dim=1)

        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.error_store = None
        engine.cfg = _Cfg()
        engine.cam_methods = {
            "saliency_map": SaliencyMapMethod(GradCamEngine._gradcam_objective)
        }
        engine.active_method_id = "saliency_map"
        engine.file_name = "sample.nii.gz"
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.SIZE = 2
        engine.STRIDE = 0
        engine.target_class = 1

        with (
            patch("src.gcd.infrastructure.xai.engine.core_engine._build_model", return_value=_Model()),
            patch("src.gcd.infrastructure.xai.runtime.model_runtime_loader.os.path.exists", return_value=True),
            patch("torch.load", return_value={}),
        ):
            engine.prepare_xai_inputs(method="saliency_map")

        self.assertEqual(engine.layers, {"input": 1})
        self.assertTrue(all(item["method"] == "saliency_map" for item in engine.patch))
        self.assertTrue(all("input_gradient" in item for item in engine.patch))

    def test_prepare_xai_inputs_collects_layers_with_forward_hooks(self) -> None:
        class _Cfg:
            model = object()
            ckpt = "checkpoint.pt"

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature = torch.nn.Conv3d(1, 2, kernel_size=1)
                self.head = torch.nn.Conv3d(2, 2, kernel_size=1)
                self.xai_layer_targets = {"feature": "feature"}

            def load_state_dict(self, _state_dict, strict=False):
                return [], []

            def forward(self, value):
                return self.head(self.feature(value))

        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.error_store = None
        engine.cfg = _Cfg()
        engine.cam_methods = {
            "gradcam": GradCamMethod(GradCamEngine._gradcam_objective)
        }
        engine.active_method_id = "gradcam"
        engine.file_name = "sample.nii.gz"
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.SIZE = 2
        engine.STRIDE = 0
        engine.target_class = 1

        with (
            patch("src.gcd.infrastructure.xai.engine.core_engine._build_model", return_value=_Model()),
            patch("src.gcd.infrastructure.xai.runtime.model_runtime_loader.os.path.exists", return_value=True),
            patch("torch.load", return_value={}),
        ):
            engine.prepare_xai_inputs(method="gradcam")

        self.assertEqual(engine.layers, {"feature": 2})
        self.assertTrue(all(item["method"] == "gradcam" for item in engine.patch))
        self.assertTrue(all("feature" in item["layers"] for item in engine.patch))
        layer_payload = engine.patch[0]["layers"]["feature"]
        self.assertIn("activation", layer_payload)
        self.assertIn("gradient", layer_payload)

    def test_prepare_xai_inputs_requires_xai_layer_targets_for_layer_methods(self) -> None:
        class _Cfg:
            model = object()
            ckpt = "checkpoint.pt"

        class _Model(torch.nn.Module):
            def load_state_dict(self, _state_dict, strict=False):
                return [], []

            def forward(self, value):
                return torch.cat([value.mean(dim=1), value.sum(dim=1)], dim=1)

        engine = GradCamEngine.__new__(GradCamEngine)
        engine._logger = lambda _message: None
        engine.error_store = None
        engine.cfg = _Cfg()
        engine.cam_methods = {
            "gradcam": GradCamMethod(GradCamEngine._gradcam_objective)
        }
        engine.active_method_id = "gradcam"
        engine.file_name = "sample.nii.gz"
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.SIZE = 2
        engine.STRIDE = 0
        engine.target_class = 1

        with (
            patch("src.gcd.infrastructure.xai.engine.core_engine._build_model", return_value=_Model()),
            patch("src.gcd.infrastructure.xai.runtime.model_runtime_loader.os.path.exists", return_value=True),
            patch("torch.load", return_value={}),
            self.assertRaisesRegex(RuntimeError, "xai_layer_targets"),
        ):
            engine.prepare_xai_inputs(method="gradcam")

    def test_dataset_input_omits_large_result_and_patch_payloads(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine.cam = torch.ones((2, 2, 2), dtype=torch.float32)
        engine.volume_data = torch.ones((2, 2, 2), dtype=torch.float32)
        engine.img0 = None
        engine.img1 = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        engine.origin_img = None
        engine.origin_meta = {}
        engine.origin_shape = (2, 2, 2)
        engine.img1_spacing = (1.0, 1.0, 1.0)
        engine.display_metadata = {}
        engine.layers = {"layer-a": 1}
        engine.file_name = "sample.nii.gz"
        engine.patch = [{"method": "gradcam", "layers": {"layer-a": torch.ones(1)}}]
        engine.target_class = 1
        engine.active_method_id = "gradcam"
        engine.model_output = torch.zeros((2, 2, 2), dtype=torch.float32)
        engine.xai_cache_key = "cfg.py|1|gradcam"

        state = engine.dataset_input()

        self.assertFalse(hasattr(state, "cam"))
        self.assertFalse(hasattr(state, "volume_data"))
        self.assertFalse(hasattr(state, "patch"))
        self.assertFalse(hasattr(state, "model_output"))
        self.assertIsNotNone(state.img1)


if __name__ == "__main__":
    unittest.main()
