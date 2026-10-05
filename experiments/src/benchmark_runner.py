"""Run a reproducible tiled CAM benchmark with any compatible GCD model.

Use ``python -m experiments.benchmark --help`` for the command line.
The config specifies the model, checkpoint, preprocessing and inference.
Numerical parity still requires the same checkpoint and cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .xai.benchmark_cam import FAMILIES, feature_layers, generate, normalize
from .xai.benchmark_metrics import (
    BEHAVIOUR_METRICS, SPATIAL_METRICS, auc, behaviour_curve, spatial_metrics, evaluate_heatmap,
)

METHODS = tuple(
    f"{family}_L{layer}" for layer in (1, 2, 3) for family in FAMILIES
) + tuple(f"multilayer_{family}" for family in FAMILIES[:3]) + (
    "prediction_logits_minmax", "prediction_mask_binary", "prediction_distance_prior",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True,
                        help="GCD MMEngine model config")
    parser.add_argument("--checkpoint", type=Path, help="Override config checkpoint")
    parser.add_argument("--case", nargs="+", action="append", metavar="ID_IMAGE_LABEL",
                        required=True, help="ID IMAGE [LABEL]; LABEL required unless --heatmap-only")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Generate CAMs without GT labels or perturbation metrics")
    parser.add_argument("--profile", choices=("gcd", "xai_hw_acdc"), default="gcd",
                        help="Output protocol label only; runtime settings come from config")
    parser.add_argument("--layers", nargs=3, metavar=("L1", "L2", "L3"),
                        help="Layer aliases/paths; defaults follow the model and benchmark config")
    parser.add_argument("--methods", nargs="+", default=["gradcam_L3"],
                        help="Method slugs, or all")
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--baseline", choices=("zero", "minimum", "mean"), default="zero")
    parser.add_argument("--label-policy", choices=("original", "masked"), default="original")
    parser.add_argument("--mode", choices=("insertion", "deletion", "both"), default="both")
    parser.add_argument("--preserve-answer", action="store_true",
                        help="Also evaluate a separate curve protecting every target GT voxel")
    parser.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--sw-batch-size", type=int)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-nifti", action="store_true",
                        help="Export original-space heatmaps and the full multiclass prediction")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _input_record(path: Path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _protocol(args, cases, checkpoint, methods, roi, sw_batch_size, blend,
              resolved_config, layer_names, cam_protocol=None):
    sources = [Path(__file__), Path(__file__).parent / "xai" / "benchmark_cam.py",
               Path(__file__).parent / "xai" / "benchmark_metrics.py",
               Path(__file__).parent / "preprocessing.py"]
    from src.gcd.infrastructure import preprocessing
    sources.append(Path(preprocessing.__file__))
    from src.gcd.infrastructure.xai.methods import benchmark_cam
    sources.append(Path(benchmark_cam.__file__))
    from src.gcd.infrastructure.xai import benchmark_metrics, evaluation
    from src.gcd.infrastructure.xai import cam_protocol as protocol_module
    sources.extend([Path(benchmark_metrics.__file__), Path(evaluation.__file__), Path(protocol_module.__file__)])
    return {
        "model_config": _input_record(args.model_config),
        "resolved_model_config_sha256": hashlib.sha256(resolved_config.encode()).hexdigest(),
        "checkpoint": _input_record(checkpoint),
        "cases": [{"id": case_id, "image": _input_record(image),
                   "label": _input_record(label) if label is not None else None}
                  for case_id, image, label in cases],
        "heatmap_only": args.heatmap_only,
        "prediction_export": {"model_space": True, "original_space": args.export_nifti,
                              "interpolation": "nearest", "labels": "all classes"},
        "profile": args.profile,
        "layers": layer_names or "last-three-Conv3d[-4:-1]",
        "methods": methods, "class_id": args.class_id, "steps": args.steps,
        "baseline": args.baseline, "label_policy": args.label_policy,
        "mode": args.mode, "preserve_answer": args.preserve_answer,
        "roi": roi, "overlap": args.overlap, "sw_batch_size": sw_batch_size,
        "blend": blend, "score_batch_size": args.score_batch_size,
        "resolved_config": resolved_config,
        "cam_protocol": protocol_module.resolve_cam_protocol(cam_protocol),
        "empty_empty_dice": 0.0, "ranking_ties": "stable input order",
        "auc": "trapezoidal 0-100%; per case then unweighted case mean",
        "implementation_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources},
    }


from src.gcd.infrastructure.xai.evaluation import prediction as _prediction, load_case, ConfiguredPredictor


def _load_case(image_path, label_path, cfg, profile, roi):
    return load_case(image_path, label_path, cfg, roi=roi)


def _export_heatmap(saliency, model_affine, source_path, output_path):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    source = nib.load(str(source_path))
    model_image = nib.Nifti1Image(saliency.astype(np.float32), model_affine)
    restored = resample_from_to(model_image, (source.shape[:3], source.affine), order=1)
    data = normalize(np.asarray(restored.dataobj, dtype=np.float32))
    if not np.isfinite(data).all():
        raise ValueError("Original-space heatmap contains non-finite values")
    header = source.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, source.affine, header), str(output_path))


def run(args):
    import torch
    from mmengine import Config
    from monai.inferers import SlidingWindowInferer

    methods = tuple(METHODS if args.methods == ["all"] else args.methods)
    if not methods or set(methods) - set(METHODS):
        raise ValueError(f"Unknown methods: {sorted(set(methods) - set(METHODS))}")
    if len(methods) != len(set(methods)):
        raise ValueError("methods must be unique")
    if args.steps < 2 or args.score_batch_size < 1:
        raise ValueError("Invalid steps, overlap, or ScoreCAM batch size")
    controls = {"prediction_logits_minmax", "prediction_mask_binary",
                "prediction_distance_prior"}
    needs_feature_layers = bool(set(methods) - controls)
    cases = []
    for case in args.case:
        if len(case) not in (2, 3) or (len(case) == 2 and not args.heatmap_only):
            raise ValueError("--case requires ID IMAGE LABEL, or ID IMAGE with --heatmap-only")
        cases.append((case[0], Path(case[1]), Path(case[2]) if len(case) == 3 else None))
    if args.heatmap_only and args.preserve_answer:
        raise ValueError("--preserve-answer requires GT evaluation")
    if len({case_id for case_id, _, _ in cases}) != len(cases) or any(
        not case_id or case_id in {".", ".."} or "/" in case_id or "\\" in case_id
        for case_id, _, _ in cases
    ):
        raise ValueError("Case IDs must be unique simple names")
    cfg = Config.fromfile(str(args.model_config))
    from src.gcd.infrastructure.xai.cam_protocol import resolve_cam_protocol
    cfg.cam_protocol = resolve_cam_protocol(cfg.get("cam_protocol"))
    checkpoint = args.checkpoint or Path(str(cfg.ckpt))
    args.overlap = cfg.inference.overlap if args.overlap is None else args.overlap
    args.device = cfg.inference.device if args.device is None else args.device
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    connected_components = bool(cfg.get("postprocessing", {}).get("keep_largest_connected_component", False))
    if args.layers:
        layer_names = tuple(args.layers)
    elif str(cfg.model.type) == "UNet":
        layer_names = ("decoder 1:input", "decoder 2", "decoder 3")
    elif needs_feature_layers and cfg.get("benchmark", {}).get("feature_layers") != "last-three-conv":
        raise ValueError("Specify --layers L1 L2 L3 for this GCD model")
    else:
        layer_names = None
    roi = tuple(args.roi or cfg.inference.roi_size)
    sw_batch_size = args.sw_batch_size or int(cfg.inference.sw_batch_size)
    blend = str(cfg.inference.blend_mode)
    if any(size < 1 for size in roi) or sw_batch_size < 1:
        raise ValueError("ROI and sliding-window batch size must be positive")
    for path in (args.model_config, checkpoint, *(p for _, image, label in cases
                                                   for p in (image, label) if p is not None)):
        if not path.is_file():
            raise FileNotFoundError(path)
    record = _protocol(args, cases, checkpoint, methods, roi, sw_batch_size, blend,
                       cfg.pretty_text, layer_names, cfg.cam_protocol)
    run_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output / f"class{args.class_id}_{args.profile}_steps{args.steps}_{run_id}"
    print(f"run={run_id} cases={len(cases)} methods={len(methods)} output={output}", flush=True)
    if args.dry_run:
        print(json.dumps(record, indent=2), flush=True)
        return output
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    cfg.ckpt = str(checkpoint)
    cfg.inference.update(dict(device=str(device), roi_size=roi, overlap=args.overlap,
                              sw_batch_size=sw_batch_size))
    predictor = ConfiguredPredictor(cfg)
    model = predictor.runtime.model
    layers = feature_layers(model, layer_names) if needs_feature_layers else ()
    logits_for = predictor.logits

    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps({"run_id": run_id, **record}, indent=2),
                                          encoding="utf-8")
    summaries = []
    for case_id, image_path, label_path in cases:
        if args.heatmap_only:
            from src.gcd.infrastructure.preprocessing import load_image
            from src.gcd.infrastructure.xai.evaluation import nifti_volume
            nifti_volume(image_path)
            image_meta, label_meta = load_image(image_path, cfg, roi=roi), None
        else:
            image_meta, label_meta = _load_case(image_path, label_path, cfg, args.profile, roi)
        image = image_meta.as_tensor().unsqueeze(0).to(device, dtype=torch.float32)
        volume = image[0, 0].detach().cpu().numpy()
        label = label_meta.as_tensor().squeeze().cpu().numpy() if label_meta is not None else None
        logits = logits_for(image)
        if not 0 <= args.class_id < logits.shape[1]:
            raise ValueError(f"class {args.class_id} is not in model output")
        full_prediction, full_probability = _prediction(logits, args.class_id,
                                                         connected_components)
        # Save the same full-volume readout used by the experiment, once per case.
        case_directory = output / case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        np.save(case_directory / "prediction_model_space.npy", full_prediction)
        if args.export_nifti:
            from src.gcd.infrastructure.xai.evaluation import export_prediction
            export_prediction(full_prediction, np.asarray(image_meta.affine), image_path,
                              case_directory / "prediction_original_space.nii.gz")

        def infer(array):
            tensor = torch.as_tensor(array, dtype=torch.float32, device=device)[None, None]
            return _prediction(logits_for(tensor), args.class_id,
                               connected_components)

        spacing = tuple(float(v) for v in cfg.preprocessing.spacing)

        for method in methods:
            directory = output / case_id / method
            summary_path, curve_path = directory / "summary.json", directory / "curve.csv"
            heatmap_path = directory / "heatmap_model_space.npy"
            nifti_path = directory / "heatmap_original_space.nii.gz"
            if not args.force and summary_path.is_file() and (args.heatmap_only or curve_path.is_file()) \
                    and heatmap_path.is_file() and (not args.export_nifti or nifti_path.is_file()):
                old = json.loads(summary_path.read_text(encoding="utf-8"))
                if old.get("run_id") == run_id:
                    print("cached", case_id, method, flush=True)
                    summaries.append(old)
                    continue
            print("running", case_id, method, flush=True)
            saliency = generate(model, image, logits, args.class_id, method, layers, roi,
                                args.overlap, args.score_batch_size, spacing,
                                blend_mode=blend,
                                sigma_scale=cfg.inference.get("sigma_scale", .125),
                                cam_protocol=cfg.get("cam_protocol"))
            if saliency.shape != volume.shape or not np.isfinite(saliency).all():
                raise RuntimeError(f"Invalid heatmap for {case_id}/{method}")
            modes = ("insertion", "deletion") if args.mode == "both" else (args.mode,)
            all_rows, evaluated = ([], {}) if args.heatmap_only else evaluate_heatmap(
                image=volume, saliency=saliency, label=label, class_id=args.class_id,
                full_prediction=full_prediction, full_probability=full_probability,
                infer=infer, spacing=spacing, steps=args.steps, baseline_mode=args.baseline,
                label_policy=args.label_policy, modes=modes,
                preserve_answer=args.preserve_answer, show_progress=True,
            )
            summary = {"run_id": run_id, "case": case_id, "method": method,
                       "layers": [name for name, _ in layers],
                       "heatmap_only": args.heatmap_only,
                       "heatmap_shape": list(saliency.shape),
                       "heatmap_min": float(saliency.min()), "heatmap_max": float(saliency.max()),
                       "target_voxels": int((full_prediction == args.class_id).sum()), **evaluated}
            directory.mkdir(parents=True, exist_ok=True)
            _csv(curve_path, all_rows)
            np.save(heatmap_path, saliency)
            if args.export_nifti:
                _export_heatmap(saliency, np.asarray(image_meta.affine), image_path, nifti_path)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries.append(summary)
    _csv(output / "metrics_by_case.csv", summaries)
    means = []
    for method in methods:
        selected = [row for row in summaries if row["method"] == method]
        numerical = [key for key, value in selected[0].items()
                     if isinstance(value, (int, float)) and not isinstance(value, bool)]
        means.append({"method": method, "case_count": len(selected),
                      **{key: float(np.mean([row[key] for row in selected]))
                         for key in numerical}})
    _csv(output / "metrics_mean.csv", means)
    print("saved", output, flush=True)
    return output


def main(argv=None):
    run(parse_args(argv))
