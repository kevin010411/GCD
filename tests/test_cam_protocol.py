import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmengine import Config

from src.gcd.infrastructure.xai.cam_protocol import resolve_cam_protocol, target_score
from src.gcd.infrastructure.xai.methods.benchmark_cam import generate, LayerTap, raw_gradient_cam
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine


class CamProtocolTests(unittest.TestCase):
    def test_cfg_and_engine_default_and_legacy(self):
        root = Path(__file__).resolve().parents[1]
        for name, region, reduction, stage in (
                ("xai_hw_acdc", "full_prediction", "mean", "per_tile"),
                ("xai_hw_acdc_legacy_ui", "tile_prediction", "sum", "after_fusion")):
            cfg = Config.fromfile(str(root / "config/model" / (name + ".py")))
            engine = GradCamEngine.__new__(GradCamEngine)
            engine.cfg = cfg
            actual = engine._configured_tile_params("gradcam", None)["cam_protocol"]
            self.assertEqual(actual, dict(target_region=region, reduction=reduction, relu_stage=stage))

    def test_unet_legacy_configs_preserve_model_and_input(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("unet_3d", "unetcnx"):
            original = Config.fromfile(str(root / "config/model" / (name + ".py")))
            legacy = Config.fromfile(str(root / "config/model" / (name + "_legacy_ui.py")))
            self.assertEqual(legacy.cam_protocol, dict(target_region="tile_prediction", reduction="sum", relu_stage="after_fusion"))
            for key in ("model", "ckpt", "preprocessing", "inference", "default_layer"):
                self.assertEqual(legacy[key], original[key])

    def test_invalid_fields_fail(self):
        for bad in ({"reduction": "max"}, {"target_region": "gt"}, {"relu_stage": "none"}, {"typo": 1}):
            with self.assertRaises(ValueError):
                resolve_cam_protocol(bad)

    def test_sum_mean_and_empty_gradients(self):
        for reduction, expected in (("sum", 1.), ("mean", .5)):
            value = torch.tensor([2., 4.], requires_grad=True)
            logits = value.reshape(1, 1, 1, 1, 2)
            mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
            target_score(logits, 0, mask, reduction).backward()
            torch.testing.assert_close(value.grad, torch.full_like(value, expected))
        empty = torch.zeros_like(mask)
        self.assertEqual(target_score(logits, 0, empty).item(), 0.)

    def test_signed_raw_cam_and_overlap_cancellation(self):
        activation = torch.tensor([-2., 4.]).reshape(1, 1, 1, 1, 2)
        gradient = torch.ones_like(activation)
        torch.testing.assert_close(raw_gradient_cam(activation, gradient, "hirescam", rectify=False), activation)
        torch.testing.assert_close(raw_gradient_cam(activation, gradient, "hirescam"), activation.relu())
        image = torch.ones(1, 1, 3, 1, 1)
        logits = torch.cat([torch.zeros_like(image), image], dim=1)
        layers = [("a", LayerTap(torch.nn.Identity()))] * 3
        def fake(*args):
            positive = args[2][0, 0, 0, 0, 0].item() == 1.
            raw = torch.tensor([1., 4.] if positive else [-6., 2.]).reshape(2, 1, 1)
            return raw.relu() if args[-1]["relu_stage"] == "per_tile" else raw
        image[0, 0, 1] = 2.
        with patch("src.gcd.infrastructure.xai.methods.benchmark_cam.patch_cam", side_effect=fake):
            before = generate(torch.nn.Identity(), image, logits, 1, "hirescam_L1", layers, (2, 1, 1), .5,
                              blend_mode="constant")
            after = generate(torch.nn.Identity(), image, logits, 1, "hirescam_L1", layers, (2, 1, 1), .5,
                             blend_mode="constant", cam_protocol={"relu_stage": "after_fusion"})
        self.assertGreater(before[1, 0, 0], 0.)
        self.assertEqual(after[1, 0, 0], 0.)

    def test_global_vs_tile_target_masks(self):
        image = torch.ones(1, 1, 3, 1, 1)
        full = torch.cat([image, -image], dim=1)  # full prediction class 0
        class Local(torch.nn.Module):
            def forward(self, value):
                return torch.cat([-value, value], dim=1)  # tile prediction class 1
        layers = [("a", LayerTap(torch.nn.Identity()))] * 3
        for region, expected in (("full_prediction", False), ("tile_prediction", True)):
            def fake(*args):
                self.assertEqual(bool(args[4].any()), expected)
                return torch.ones(2, 1, 1)
            with patch("src.gcd.infrastructure.xai.methods.benchmark_cam.patch_cam", side_effect=fake):
                generate(Local(), image, full, 1, "gradcam_L1", layers, (2, 1, 1), .5,
                         cam_protocol={"target_region": region})
