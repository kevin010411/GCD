"""Opt-in real ACDC acceptance: GCD_RUN_REAL_CAM_PARITY=1 inside Main.

Runs the actual UI backend with patient103, original xai_hw weights and L3.
The JSON report records heatmap errors and dense21 insertion/deletion AUC.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer

from experiments.src.preprocessing import load_image
from experiments.src.xai.benchmark_cam import feature_layers, generate
from experiments.src.xai import benchmark_metrics
from experiments.src.benchmark_runner import _prediction
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine, _build_model


@unittest.skipUnless(os.environ.get("GCD_RUN_REAL_CAM_PARITY") == "1", "opt-in real GPU/data test")
class RealCamBenchmarkParityTests(unittest.TestCase):
    def test_patient103_l3_four_methods_and_gradcam_dense21(self):
        torch.set_num_threads(4)
        root = Path(__file__).resolve().parents[1]
        colab = Path("/workspace/xai_hw/assignment/colab")
        checkpoint = colab / "checkpoint/best_model.pth"
        image_path = colab / "data/testing/patient103/patient103_frame01.nii.gz"
        label_path = colab / "teacher/final_labels/patient103/patient103_frame01_gt.nii.gz"
        self.assertTrue(checkpoint.is_file() and image_path.is_file() and label_path.is_file())
        sys.path.insert(0, str(colab))
        from benchmark_suite import methods as reference_methods
        engine = GradCamEngine(str(root / "config/model/xai_hw_acdc.py"))
        engine.cfg.ckpt = str(checkpoint)
        cfg = engine.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(cfg.model).to(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
        model.eval()
        image = load_image(image_path, cfg)
        label = load_image(label_path, cfg, label=True).as_tensor().squeeze().numpy()
        batch = image.as_tensor()[None].to(device)
        inferer = SlidingWindowInferer(roi_size=tuple(cfg.inference.roi_size),
                                      sw_batch_size=cfg.inference.sw_batch_size,
                                      overlap=cfg.inference.overlap, mode=cfg.inference.blend_mode)
        with torch.inference_mode():
            logits = inferer(batch, model)
        layers = feature_layers(model)
        layer = layers[2][0]
        engine.load_volume(str(image_path))
        engine.set_target_class(1)
        report = {"case": "patient103", "class": 1, "layer": layer,
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "roi": list(cfg.inference.roi_size), "overlap": cfg.inference.overlap,
                  "blend": cfg.inference.blend_mode, "methods": {}}
        gradcam_maps = None
        for family in ("gradcam", "hirescam", "layercam", "scorecam"):
            print("real parity:", family, flush=True)
            expected = generate(model, batch, logits, 1, family + "_L3", layers,
                                tuple(cfg.inference.roi_size), cfg.inference.overlap,
                                blend_mode=cfg.inference.blend_mode)
            reference, _ = reference_methods.generate(family + "_L3", model, batch, 1, device,
                                                      logits, logits.argmax(1)[0].cpu().numpy())
            params = {"_selected_layer": layer, "_feature_start": 0, "_feature_stop": 999}
            engine.prepare_xai_inputs(method=family, objective_id="predicted_target_mask", method_params=params)
            tile_count = len(engine.tile_plan.regions)
            engine.run_xai_method(method=family, layer=layer, method_params=params)
            inverse = tuple(int(v) for v in np.argsort(engine.PERMUTE))
            actual = engine.cam.permute(*inverse).numpy().copy()
            difference = np.abs(actual - expected)
            report["methods"][family] = {
                "ui_benchmark_max_abs": float(difference.max()),
                "ui_benchmark_mean_abs": float(difference.mean()),
                "benchmark_xai_hw_max_abs": float(np.abs(expected - reference).max()),
                "tile_count": tile_count,
            }
            np.testing.assert_allclose(engine.img1.as_tensor().numpy(), image.as_tensor().numpy(), atol=0, rtol=0)
            # CPU vs CUDA accumulation/reductions can differ slightly.
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(expected, reference, atol=1e-4, rtol=1e-4)
            if family == "gradcam":
                gradcam_maps = actual, expected
        full_prediction, full_probability = _prediction(logits, 1, True)
        cache = {}

        def infer(array):
            key = hashlib.sha256(np.asarray(array, np.float32).tobytes()).digest()
            if key not in cache:
                with torch.inference_mode():
                    output = inferer(torch.as_tensor(array, device=device)[None, None], model)
                cache[key] = _prediction(output, 1, True)
            return cache[key]

        kwargs = dict(image=batch[0, 0].cpu().numpy(), label=label, class_id=1,
                      full_prediction=full_prediction, full_probability=full_probability,
                      infer=infer, steps=21, baseline_mode="zero", label_policy="original",
                      modes=("insertion", "deletion"))
        ui_rows, _ = benchmark_metrics.behaviour_curve(saliency=gradcam_maps[0], **kwargs)
        batch_rows, _ = benchmark_metrics.behaviour_curve(saliency=gradcam_maps[1], **kwargs)
        report["gradcam_dense21_auc"] = {}
        for mode in kwargs["modes"]:
            for metric in benchmark_metrics.BEHAVIOUR_METRICS:
                ui_auc = benchmark_metrics.auc(ui_rows, metric, mode)
                batch_auc = benchmark_metrics.auc(batch_rows, metric, mode)
                report["gradcam_dense21_auc"][mode + "_" + metric] = {
                    "ui": ui_auc, "benchmark": batch_auc, "difference": abs(ui_auc - batch_auc)}
        report["distinct_perturbed_inputs"] = len(cache)
        destination = Path(os.environ.get("GCD_PARITY_REPORT", "/workspace/gcd_ui_benchmark_parity_results.json"))
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        for record in report["gradcam_dense21_auc"].values():
            self.assertLessEqual(record["difference"], 1e-5)


if __name__ == "__main__":
    unittest.main()
