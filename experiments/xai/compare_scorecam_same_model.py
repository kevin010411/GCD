"""Compare GCD Score-CAM with the xai_hw formula on one shared UNet3D tile.

This is a diagnostic adapter: the reference formula is transcribed from
assignment/colab/benchmark_baselines.py::RawScoreCAM. Both paths receive the
same GCD checkpoint, preprocessed CT, decoder layer, and fixed target mask.
"""

from __future__ import annotations

import json
from pathlib import Path

print("loading dependencies", flush=True)
import nibabel as nib
import numpy as np
import platformdirs

platformdirs.user_cache_dir = lambda *_args, **_kwargs: str(
    Path("output/scorecam_tmp").resolve()
)

import torch
import torch.nn.functional as F
from mmengine import Config

print("loading GCD", flush=True)
import src.model  # noqa: F401
from experiments.runner import _load_image
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
from src.gcd.infrastructure.xai.methods.base import CamPatchContext
from src.gcd.infrastructure.xai.methods.scorecam import ScoreCamMethod
from src.gcd.infrastructure.xai.runtime.layer_hooks import XaiLayerHookManager
from src.gcd.infrastructure.xai.runners.xai_cam_runner import XaiCamRunRequest, XaiCamRunner
from src.gcd.infrastructure.xai.tiling.scorecam_blend import gaussian_importance_map
from src.gcd.infrastructure.xai.tiling.tile_collector import TileCollectionRequest, TileCollector
from src.gcd.infrastructure.xai.tiling.tile_strategy import SlidingWindowTileStrategy
from src.utils import build_model


def reference_scorecam(model, layer, window, class_id, target_mask, batch_size=2):
    """xai_hw RawScoreCAM equations, with masked forwards batched for memory."""
    captured = {}
    hook = layer.register_forward_hook(
        lambda _module, _inputs, output: captured.update(value=output.detach())
    )
    try:
        with torch.inference_mode():
            original_logits = model(window)
    finally:
        hook.remove()
    activation = captured["value"][0].relu()
    resized = F.interpolate(
        activation[None], size=window.shape[-3:], mode="trilinear", align_corners=False
    )[0]
    flattened = resized.flatten(start_dim=1)
    minimum = flattened.min(dim=1).values[:, None, None, None]
    maximum = flattened.max(dim=1).values[:, None, None, None]
    masks = (resized - minimum) / (maximum - minimum).clamp_min(1e-8)
    masked_inputs = window * masks[:, None]
    target_mask_float = target_mask.to(dtype=window.dtype)
    target_count = target_mask_float.sum().clamp_min(1.0)
    scores = []
    with torch.inference_mode():
        for batch in masked_inputs.split(batch_size, dim=0):
            output = model(batch)
            scores.append(
                (output[:, class_id] * target_mask_float).sum(dim=(-3, -2, -1))
                / target_count
            )
    scores = torch.cat(scores)
    weights = torch.softmax(scores, dim=0)
    cam = (weights[:, None, None, None] * activation).sum(dim=0).relu()
    cam = F.interpolate(
        cam[None, None], size=window.shape[-3:], mode="trilinear", align_corners=False
    )[0, 0]
    return dict(
        original_logits=original_logits.detach(),
        activation=activation.detach(),
        scores=scores.detach(),
        weights=weights.detach(),
        cam=cam.detach(),
        constant_channels=((maximum - minimum).flatten() <= 1e-12).detach(),
    )


def diff(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return {
        "max_abs": float((a - b).abs().max()),
        "mean_abs": float((a - b).abs().mean()),
        "pearson": float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1]),
    }


def main():
    print("reading config", flush=True)
    cfg = Config.fromfile("experiments/configs/scorecam_unet3d_audit.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    print("reading checkpoint", flush=True)
    checkpoint = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    model.eval()
    print("preprocessing CT", flush=True)
    image_path = Path("data/chgh/patient0016.nii.gz")
    image = _load_image(image_path, cfg).to(device=device, dtype=torch.float32)
    prediction_path = Path(
        "output/scorecam_unet3d_xai_hw_style_patient0016/patient0016_prediction.nii.gz"
    )
    prediction = np.asarray(nib.load(str(prediction_path)).dataobj)
    assert tuple(prediction.shape) == tuple(image.shape[-3:])
    target_positions = np.argwhere(prediction == 1)
    center = np.median(target_positions, axis=0).astype(int)
    size = 128
    starts = [max(0, min(int(c) - size // 2, int(n) - size)) for c, n in zip(center, image.shape[-3:])]
    spatial = tuple(slice(start, start + size) for start in starts)
    window = image[(slice(None), *spatial)].unsqueeze(0).detach()
    fixed_target = torch.as_tensor(prediction[spatial] == 1, device=device)
    layer_name = "decoder 2"
    layer = model.get_submodule(model.xai_layer_targets[layer_name])
    print(f"device={device} image={tuple(image.shape)} tile={starts} target={int(fixed_target.sum())}", flush=True)
    assert fixed_target.any()

    reference = reference_scorecam(model, layer, window, 1, fixed_target)
    print("reference complete", flush=True)
    with XaiLayerHookManager(model, selected_layers=(layer_name,)) as hooks:
        with torch.no_grad():
            logits = model(window)
        activation = hooks.layers_by_name()[layer_name]
        method = ScoreCamMethod(GradCamEngine._predicted_target_mask_objective)
        payload = method.collect_patch_data(
            CamPatchContext(
                input_tensor=window,
                logits=logits,
                layers_by_name={layer_name: activation},
                target_class=1,
                objective=GradCamEngine._predicted_target_mask_objective,
                model=model,
                method_params={
                    "_selected_layer": layer_name,
                    "_objective_id": "predicted_target_mask",
                    "_fixed_target_mask": fixed_target[None],
                },
                device=device,
            )
        )
    gcd_cam = F.interpolate(
        payload["cam"].float(), size=window.shape[-3:], mode="trilinear", align_corners=False
    )[0, 0]
    valid = payload["valid_channels"].bool()
    result = {
        "checkpoint": str(cfg.ckpt),
        "image": str(image_path),
        "layer": layer_name,
        "class": 1,
        "tile_origin": starts,
        "tile_size": size,
        "fixed_target_voxels": int(fixed_target.sum()),
        "channels": int(valid.numel()),
        "activation_shape": list(activation.shape),
        "reference_constant_channels": int(reference["constant_channels"].sum()),
        "gcd_valid_channels": int(valid.sum()),
        "maximum_feature_weight": float(reference["weights"].max()),
        "effective_feature_count": float(1.0 / reference["weights"].square().sum()),
        "original_logits": diff(logits, reference["original_logits"]),
        "activation": diff(activation.relu(), reference["activation"][None]),
        "scores_valid_only": diff(payload["scores"][valid], reference["scores"][valid]),
        "weights": diff(payload["weights"], reference["weights"]),
        "heatmap": diff(gcd_cam, reference["cam"]),
        "reference_weight_on_constant_channels": float(
            reference["weights"][reference["constant_channels"]].sum()
        ),
        "score_range_reference": [float(reference["scores"].min()), float(reference["scores"].max())],
        "score_range_gcd_valid": [float(payload["scores"][valid].min()), float(payload["scores"][valid].max())],
    }
    # A real CT subvolume with two overlapping 128-cubed tiles checks the
    # fixed full-prediction target and Gaussian fusion without a second
    # whole-patient run of 2,048 masked forwards.
    volume_start = [max(0, starts[0] - 16), starts[1], starts[2]]
    volume_shape = (160, 128, 128)
    volume_slices = tuple(
        slice(start, start + length)
        for start, length in zip(volume_start, volume_shape)
    )
    volume = image[(slice(None), *volume_slices)].detach()
    plan = SlidingWindowTileStrategy().plan(
        input_shape=volume_shape, patch_size=128, stride=96
    )
    assert len(plan.regions) == 2
    method_params = {
        "_selected_layer": layer_name,
        "_objective_id": "predicted_target_mask",
        "_feature_start": 0,
        "_feature_stop": 64,
    }
    request = TileCollectionRequest(
        model=model,
        device=device,
        model_input=volume,
        method=method,
        objective=GradCamEngine._predicted_target_mask_objective,
        target_class=1,
        tile_plan=plan,
        method_params=method_params,
    )
    target = TileCollector._scorecam_global_target_mask(request, volume[None])
    reference_logits = torch.zeros((1, 4, *volume_shape))
    with torch.inference_mode():
        for region in plan.regions:
            tile = volume[(slice(None), *region.slices)].unsqueeze(0)
            reference_logits[(slice(None), slice(None), *region.slices)] += (
                model(tile).float().cpu()
                * gaussian_importance_map(region.size)
            )
    reference_target = reference_logits.argmax(dim=1)[0] == 1
    collection = TileCollector().collect(request)
    gcd_fused = XaiCamRunner().run(
        XaiCamRunRequest(
            method=method,
            patches=collection.patches,
            layers=collection.layers,
            img1=volume.cpu(),
            size=128,
            stride=96,
            permute=(0, 1, 2),
            default_layer=layer_name,
            tile_plan=plan,
            layer=layer_name,
        )
    ).cam
    accum = torch.zeros(volume_shape)
    coverage = torch.zeros(volume_shape)
    tile_diffs = []
    for region, patch in zip(plan.regions, collection.patches):
        tile = volume[(slice(None), *region.slices)].unsqueeze(0)
        tile_target = target[region.slices].to(device)
        ref = reference_scorecam(model, layer, tile, 1, tile_target)
        weight = gaussian_importance_map(region.size)
        accum[region.slices] += ref["cam"].cpu() * weight
        coverage[region.slices] += weight
        gcd_tile = F.interpolate(
            patch["cam"].float(),
            size=region.size,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        tile_diffs.append(diff(gcd_tile, ref["cam"]))
    reference_fused = (accum / coverage).clamp_min(0)
    reference_fused -= reference_fused.min()
    reference_fused /= reference_fused.max().clamp_min(1e-12)
    result["two_tile_fusion"] = {
        "tile_count": len(plan.regions),
        "global_target_voxels": int(target.sum()),
        "global_target_mask_disagreements": int((target != reference_target).sum()),
        "gcd_masked_forwards": sum(int(p["masked_forward_count"]) for p in collection.patches),
        "tile_heatmaps": tile_diffs,
        "fused_heatmap": diff(gcd_fused, reference_fused),
    }
    output_dir = Path("output/scorecam_same_model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "patient0016_decoder2_class1.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
