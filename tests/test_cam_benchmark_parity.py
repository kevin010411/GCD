"""UI tile collection/runner must match the shared benchmark CAM protocol."""
import unittest
import gc
import weakref
from itertools import product
import numpy as np
import torch

from experiments.src.xai.benchmark_cam import feature_layers, generate
from src.gcd.infrastructure.xai.methods import XaiMethodRegistry
from src.gcd.infrastructure.xai.tiling.tile_strategy import SlidingWindowTileStrategy
from src.gcd.infrastructure.xai.tiling.tile_collector import TileCollector, TileCollectionRequest
from src.gcd.infrastructure.xai.runners.xai_cam_runner import XaiCamRunner, XaiCamRunRequest
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv3d(1, 3, 1)
        self.b = torch.nn.Conv3d(3, 3, 1)
        self.c = torch.nn.Conv3d(3, 3, 1)
        self.head = torch.nn.Conv3d(3, 2, 1)
        with torch.no_grad():
            self.a.weight[:, 0, 0, 0, 0] = torch.tensor([1., -1., 0.])
            self.a.bias[:] = torch.tensor([0., 0., 2.])
            for module in (self.b, self.c):
                module.weight.zero_()
                module.weight[:, :, 0, 0, 0].copy_(torch.eye(3))
                module.bias.zero_()
            self.head.weight.zero_()
            self.head.weight[1, 0, 0, 0, 0] = 1.
            self.head.bias[:] = torch.tensor([0., -.15])
        self.xai_layer_targets = {name: name for name in ("a", "b", "c")}

    def forward(self, value):
        return self.head(self.c(self.b(self.a(value))))


class CamBenchmarkParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_gradient_capture_releases_activation_without_cyclic_gc(self):
        from src.gcd.infrastructure.xai.methods.benchmark_cam import _capture, LayerTap
        model = ToyModel().eval()
        references = []
        handle = model.a.register_forward_hook(lambda module, inputs, output: references.append(weakref.ref(output)))
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            captured = _capture(model, LayerTap(model.a), torch.ones(1, 1, 2, 2, 2),
                                gradient=True, class_id=1,
                                target=torch.ones(2, 2, 2, dtype=torch.bool))
            del captured
            self.assertIsNone(references[-1]())
        finally:
            handle.remove()
            if was_enabled:
                gc.enable()

    def test_all_cam_families_and_layers_match_on_rectangular_overlapping_tiles(self):
        from monai.inferers import SlidingWindowInferer
        model = ToyModel().eval()
        image = torch.linspace(-1, 1, 5 * 6 * 4).reshape(1, 1, 5, 6, 4)
        roi = (4, 4, 4)
        registry = XaiMethodRegistry.default(GradCamEngine._predicted_target_mask_objective)
        plan = SlidingWindowTileStrategy().plan(input_shape=image.shape[-3:], patch_size=roi, stride=3)
        layers = feature_layers(model)
        for blend, region, reduction, stage in product(
                ("constant", "gaussian"), ("full_prediction", "tile_prediction"),
                ("mean", "sum"), ("per_tile", "after_fusion")):
            protocol = dict(target_region=region, reduction=reduction, relu_stage=stage)
            with torch.inference_mode():
                logits = SlidingWindowInferer(roi, sw_batch_size=2, overlap=.25, mode=blend)(image, model)
            for family in ("gradcam", "hirescam", "layercam", "scorecam"):
                for index, (name, _) in enumerate(layers, 1):
                    with self.subTest(blend=blend, family=family, layer=name, protocol=protocol):
                        expected = generate(model, image, logits, 1, f"{family}_L{index}", layers, roi, .25,
                                            score_batch_size=1, blend_mode=blend, cam_protocol=protocol)
                        params = {"_selected_layer": name, "_objective_id": "predicted_target_mask",
                                  "_blend_mode": blend, "_sw_batch_size": 2, "_overlap": .25,
                                  "cam_protocol": protocol}
                        method = registry.resolve(family)
                        collected = TileCollector().collect(TileCollectionRequest(
                            model=model, device=torch.device("cpu"), model_input=image[0], method=method,
                            objective=GradCamEngine._predicted_target_mask_objective, target_class=1,
                            tile_plan=plan, method_params=params))
                        result = XaiCamRunner().run(XaiCamRunRequest(
                            method=method, patches=collected.patches, layers=collected.layers,
                            img1=image[0], size=4, stride=3, permute=(0, 1, 2), default_layer=name,
                            tile_plan=plan, layer=name, method_params=params))
                        np.testing.assert_allclose(result.cam.numpy(), expected, atol=2e-6, rtol=2e-5)
                        if family == "scorecam":
                            # Constant positive channel must take part in softmax.
                            self.assertTrue(any(p["valid_channels"][2] for p in collected.patches))

    def test_fixed_target_uses_mean_and_empty_target_has_zero_gradients(self):
        from src.gcd.infrastructure.xai.methods.base import CamPatchContext
        for empty in (False, True):
            value = torch.tensor([-2., 3., 5., 7.]).reshape(1, 1, 1, 2, 2).requires_grad_()
            value.retain_grad()
            logits = torch.cat([torch.zeros_like(value), value], dim=1)
            target = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
            if not empty:
                target[0, 0, 0, 0] = True  # Negative logit: cannot use local argmax.
                target[0, 0, 1, 1] = True
            payload = XaiMethodRegistry.default().resolve("gradcam").collect_patch_data(CamPatchContext(
                input_tensor=value, logits=logits, layers_by_name={"feature": value}, target_class=1,
                objective=lambda *args: self.fail("must use fixed mask instead"),
                method_params={"_objective_id": "predicted_target_mask", "_fixed_target_mask": target}))
            expected = target[:, None].float() / 2 if not empty else torch.zeros_like(value)
            torch.testing.assert_close(payload["layers"]["feature"]["gradient"], expected)


if __name__ == "__main__":
    unittest.main()
