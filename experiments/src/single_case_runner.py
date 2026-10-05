"""Evaluate one NIfTI heatmap against its image and segmentation label."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from mmengine import Config, DictAction

from src.gcd.infrastructure.xai.benchmark_metrics import BEHAVIOUR_METRICS, evaluate_heatmap
from src.gcd.infrastructure.xai.evaluation import ConfiguredPredictor, load_case, load_heatmap


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", "--ground-truth", dest="label", type=Path, required=True)
    parser.add_argument("--heatmap", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--heatmap-space", choices=("original", "model", "world"), default="original",
                        help="original/model strictly check shape+affine; world explicitly resamples by affine")
    parser.add_argument("--steps", type=int, default=21,
                        help="21 uses dense early points; AUC always covers 0-100 percent")
    parser.add_argument("--mode", choices=("insertion", "deletion", "both"), default="both")
    parser.add_argument("--baseline", choices=("zero", "minimum", "mean"), default="zero")
    parser.add_argument("--label-policy", choices=("original", "masked"), default="original")
    parser.add_argument("--preserve-answer", action="store_true",
                        help="Also evaluate a separate variant protecting every target GT voxel")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False), encoding="utf-8")


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_clean(rows))


def _record(path):
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _plots(directory, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(exist_ok=True)
    for metric in BEHAVIOUR_METRICS:
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = sorted({(row["variant"], row["mode"]) for row in rows})
        for variant, mode in groups:
            selected = [row for row in rows if (row["variant"], row["mode"]) == (variant, mode)]
            ax.plot([row["proportion"] * 100 for row in selected],
                    [row[metric] for row in selected], label=f"{variant}/{mode}")
        ax.set(xlabel="Perturbed fraction (%)", ylabel=metric, xlim=(0, 100))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f"{metric}.png", dpi=150)
        plt.close(fig)


def run(args):
    if args.steps < 2 or args.class_id < 0:
        raise ValueError("steps must be >=2 and class-id must be nonnegative")
    cfg = Config.fromfile(str(args.model_config))
    cfg.merge_from_dict(args.cfg_options)
    if args.checkpoint:
        cfg.ckpt = str(args.checkpoint)
    if args.device:
        cfg.inference.device = args.device
    import torch
    requested_device = str(cfg.inference.device)
    if requested_device == "auto":
        cfg.inference.device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(cfg.inference.roi_size) != 3 or any(v < 1 for v in cfg.inference.roi_size):
        raise ValueError("inference.roi_size must be three positive integers")
    if not 0 <= cfg.inference.overlap < 1 or cfg.inference.sw_batch_size < 1:
        raise ValueError("Invalid inference overlap/batch size")
    if cfg.inference.get("tile_strategy", "sliding_window") != "sliding_window":
        raise ValueError("Only sliding_window is supported")
    image_meta, label_meta = load_case(args.image, args.label, cfg)
    saliency = load_heatmap(args.heatmap, args.image, image_meta, args.heatmap_space)
    from src.gcd.infrastructure.xai import benchmark_metrics, evaluation
    record = {
        "image": _record(args.image), "label": _record(args.label),
        "heatmap": _record(args.heatmap), "model_config": _record(args.model_config),
        "checkpoint": _record(Path(cfg.ckpt)), "resolved_config": cfg.pretty_text,
        "requested_device": requested_device,
        "class_id": args.class_id, "steps": args.steps, "mode": args.mode,
        "baseline": args.baseline, "label_policy": args.label_policy,
        "preserve_answer": args.preserve_answer, "heatmap_space": args.heatmap_space,
        "heatmap_resampling": "world-affine linear interpolation then global minmax",
        "model_shape": list(saliency.shape), "model_affine": np.asarray(image_meta.affine).tolist(),
        "auc": "trapezoidal over 0-100%; dense21 for steps=21; no clipping of relative scores",
        "undefined_metrics": "null; empty-empty binary Dice=0; undefined ratios are NOT zero",
        "ranking_ties": "stable model-grid input order",
        "implementation": [_record(Path(path)) for path in (
            __file__, benchmark_metrics.__file__, evaluation.__file__,
            sys.modules["src.gcd.infrastructure.preprocessing"].__file__,
            sys.modules["src.gcd.infrastructure.xai.runtime.model_runtime_loader"].__file__,
        )],
    }
    run_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output / f"class{args.class_id}_{run_id}"
    print(f"run={run_id} output={output}", flush=True)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return output
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; use --force to replace this exact protocol")
    predictor = ConfiguredPredictor(cfg)
    volume = image_meta.as_tensor()[0].cpu().numpy()
    label = label_meta.as_tensor()[0].cpu().numpy()
    full_prediction, full_probability = predictor(volume, args.class_id)
    modes = ("insertion", "deletion") if args.mode == "both" else (args.mode,)
    rows, summary = evaluate_heatmap(
        image=volume, saliency=saliency, label=label, class_id=args.class_id,
        full_prediction=full_prediction, full_probability=full_probability,
        infer=lambda array: predictor(array, args.class_id),
        spacing=tuple(float(v) for v in image_meta.pixdim),
        steps=args.steps, baseline_mode=args.baseline,
        label_policy=args.label_policy, modes=modes,
        preserve_answer=args.preserve_answer, show_progress=True,
    )
    summary.update({"run_id": run_id, "class_id": args.class_id,
                    "constant_heatmap": bool(np.ptp(saliency) <= np.finfo(np.float32).eps)})
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "protocol.json", {
        "run_id": run_id, **record, "actual_device": str(predictor.runtime.device),
    })
    _write_json(output / "summary.json", summary)
    _csv(output / "summary.csv", [summary])
    _csv(output / "curve.csv", rows)
    import nibabel as nib
    nib.save(nib.Nifti1Image(saliency, np.asarray(image_meta.affine)),
             str(output / "heatmap_model_space.nii.gz"))
    if not args.no_plots:
        _plots(output / "plots", rows)
    print("saved", output, flush=True)
    return output


def main(argv=None):
    run(parse_args(argv))
