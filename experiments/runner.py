from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from .metrics import segmentation_metrics
from .outputs import export_csv_files, export_faithfulness_plots
from .xai import (
    compute_attribution,
    configured_perturbations,
    target_class_from_prediction,
)


def _device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{name}', but CUDA is unavailable")
    return torch.device(name)


def _load_image(path: Path, cfg: Any, *, label: bool = False):
    import monai.transforms as mt

    image = mt.LoadImage(image_only=True)(str(path))
    image = mt.EnsureChannelFirst()(image)
    image = mt.Spacing(
        pixdim=tuple(cfg.preprocessing.spacing),
        mode="nearest" if label else "bilinear",
    )(image)
    if not label:
        image = mt.ScaleIntensityRange(
            a_min=float(cfg.preprocessing.intensity_input_range[0]),
            a_max=float(cfg.preprocessing.intensity_input_range[1]),
            b_min=float(cfg.preprocessing.intensity_output_range[0]),
            b_max=float(cfg.preprocessing.intensity_output_range[1]),
            clip=True,
        )(image)
    return image


def _extract_logits(output: Any):
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Model returned an empty output sequence")
        output = output[0]
    if not hasattr(output, "ndim") or output.ndim != 5:
        raise RuntimeError(
            f"Expected model logits shaped [B, C, X, Y, Z], got {type(output)!r}"
        )
    return output


def _synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def _configured_values(config: Any, singular: str, plural: str) -> list[Any]:
    values = config.get(singular, config.get(plural, ()))
    if isinstance(values, (str, int)):
        return [values]
    return list(values or ())


def _run_single(
    args: argparse.Namespace,
    *,
    display_progress: bool = True,
    display_perturbation_progress: bool = True,
) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np
    import torch
    from mmengine import Config
    from monai.inferers import sliding_window_inference
    from tqdm.auto import tqdm

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.ground_truth is not None and not args.ground_truth.is_file():
        raise FileNotFoundError(f"Ground truth not found: {args.ground_truth}")
    if args.output.exists() and not args.output.is_dir():
        raise NotADirectoryError(f"--output must be a result directory: {args.output}")
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    case_name = _nifti_stem(args.input)
    prediction_output = output_dir / f"{case_name}_prediction.nii.gz"

    stages = tqdm(
        total=5,
        desc="Experiment",
        unit="stage",
        disable=not display_progress,
    )
    stages.set_postfix_str("loading config/model")
    cfg = Config.fromfile(str(args.config))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    import src.model  # noqa: F401 -- registers GCD models
    from src.utils import build_model

    device = _device(str(cfg.inference.device))
    model = build_model(cfg.model).to(device)
    checkpoint = Path(str(cfg.ckpt))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = (
        payload["state_dict"]
        if isinstance(payload, dict) and "state_dict" in payload
        else payload
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    stages.update()

    stages.set_postfix_str("preprocessing volume")
    image = _load_image(args.input, cfg).to(device=device, dtype=torch.float32)
    batch = image.unsqueeze(0)
    stages.update()

    def predictor(tile):
        return _extract_logits(model(tile))

    def infer(input_batch=None):
        return sliding_window_inference(
            batch if input_batch is None else input_batch,
            roi_size=tuple(cfg.inference.roi_size),
            sw_batch_size=int(cfg.inference.sw_batch_size),
            predictor=predictor,
            overlap=float(cfg.inference.overlap),
            mode=str(cfg.inference.blend_mode),
        )

    stages.set_postfix_str("inference benchmark")
    with torch.inference_mode():
        for _ in tqdm(range(int(cfg.inference.warmup_runs)), desc="Warmup", unit="run", leave=False, disable=not display_progress):
            infer()
        _synchronize(device)
        durations = []
        logits = None
        for _ in tqdm(range(int(cfg.inference.benchmark_runs)), desc="Benchmark", unit="run", leave=False, disable=not display_progress):
            started = time.perf_counter()
            logits = infer()
            _synchronize(device)
            durations.append(time.perf_counter() - started)
    stages.update()
    if logits is None:
        raise ValueError("inference.benchmark_runs must be at least 1")

    prediction = torch.argmax(logits, dim=1)[0].cpu()
    affine = np.asarray(getattr(image, "affine", np.eye(4)), dtype=np.float64)
    nib.save(nib.Nifti1Image(prediction.numpy().astype(np.int16), affine), prediction_output)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    class_count = int(logits.shape[1])
    class_voxels = {
        str(class_id): int((prediction == class_id).sum())
        for class_id in range(class_count)
    }
    metrics: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "output_directory": str(output_dir.resolve()),
        "output": str(prediction_output.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "device": str(device),
        "input_shape": list(batch.shape),
        "output_shape": list(prediction.shape),
        "class_count": class_count,
        "class_voxels": class_voxels,
        "benchmark": {
            "runs": len(durations),
            "latency_seconds": durations,
            "mean_latency_seconds": sum(durations) / len(durations),
            "volumes_per_second": len(durations) / sum(durations),
            "parameter_count": parameters,
            "parameter_bytes": parameter_bytes,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "prediction_bytes": prediction_output.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    target = None
    if args.ground_truth is not None:
        target = _load_image(args.ground_truth, cfg, label=True)[0].cpu()
        metrics["ground_truth"] = str(args.ground_truth.resolve())
        metrics["segmentation"] = segmentation_metrics(
            prediction, target, class_count, cfg
        )

    perturbations = list(configured_perturbations(cfg))
    if bool(cfg.xai.enabled) and perturbations:
        stages.set_postfix_str("XAI and faithfulness")
        xai_results: dict[str, Any] = {}
        work: dict[tuple[str, int], list[Any]] = {}
        for perturbation in perturbations:
            methods = _configured_values(perturbation.config, "method", "methods")
            classes = _configured_values(
                perturbation.config, "target_class", "target_classes"
            )
            for configured_class in classes:
                target_class = target_class_from_prediction(
                    prediction, configured_class
                )
                if not 0 <= target_class < class_count:
                    raise ValueError(
                        f"{perturbation.__class__.__name__}.target_class="
                        f"{target_class} is outside [0, {class_count - 1}]"
                    )
                for configured_method in methods:
                    key = (str(configured_method), target_class)
                    pair_metrics = work.setdefault(key, [])
                    if perturbation not in pair_metrics:
                        pair_metrics.append(perturbation)

        perturbation_step_total = 0
        for (_, target_class), metrics_for_pair in work.items():
            answer_mask = (
                (target == target_class).to(device) if target is not None else None
            )
            perturbation_step_total += sum(
                perturbation.progress_steps(answer_mask)
                for perturbation in metrics_for_pair
            )
        perturbation_progress = tqdm(
            total=perturbation_step_total,
            desc=f"{case_name} perturbations",
            unit="step",
            leave=False,
            disable=not display_perturbation_progress,
        )

        for (method, target_class), metrics_for_pair in tqdm(
            work.items(),
            desc="XAI method/classes",
            unit="pair",
            leave=False,
            disable=not display_progress,
        ):
            target_mask = (prediction == target_class).to(device)
            attribution = compute_attribution(
                batch, model, predictor, method, target_class, target_mask, cfg
            )
            result_id = f"{method}_class_{target_class}"
            xai_output = output_dir / f"{case_name}_{result_id}_xai.nii.gz"
            nib.save(
                nib.Nifti1Image(attribution.cpu().numpy().astype(np.float32), affine),
                xai_output,
            )
            result = {
                "method": method,
                "attribution": str(xai_output.resolve()),
                "target_class": target_class,
                "attribution_min": float(attribution.min()),
                "attribution_max": float(attribution.max()),
                "perturbations": {},
            }
            answer_mask = (
                (target == target_class).to(device) if target is not None else None
            )
            for perturbation in metrics_for_pair:
                result["perturbations"][perturbation.result_id] = perturbation.evaluate(
                    batch,
                    attribution.to(device),
                    infer,
                    target_class,
                    target_mask,
                    answer_mask,
                    progress=False,
                    progress_callback=perturbation_progress.update,
                )
            xai_results[result_id] = result
        perturbation_progress.close()
        metrics["xai"] = xai_results
    stages.update()

    stages.set_postfix_str("writing JSON/CSV/plots")
    metrics_output = output_dir / (args.metrics_output.name if args.metrics_output else "metrics.json")
    metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_outputs = export_csv_files(metrics, output_dir)
    plot_outputs = export_faithfulness_plots(metrics, output_dir)
    stages.update()
    stages.set_postfix_str("complete")
    stages.close()
    if display_progress:
        print(f"Complete: {case_name}")
        print(f"Result directory: {output_dir}")
        print(f"Metrics: {metrics_output}")
        print(f"CSV files: {len(csv_outputs)}, plots: {len(plot_outputs)}")
    return metrics


def _dataset_cases(input_dir: Path) -> list[tuple[Path, Path | None]]:
    volumes = sorted(
        path
        for pattern in ("*.nii", "*.nii.gz")
        for path in input_dir.glob(pattern)
        if not _nifti_stem(path).endswith("_gt")
    )
    cases = []
    seen_stems: set[str] = set()
    for volume in volumes:
        stem = _nifti_stem(volume)
        if stem in seen_stems:
            raise ValueError(
                f"Dataset contains more than one input with stem {stem!r}"
            )
        seen_stems.add(stem)
        ground_truth = next(
            (
                candidate
                for candidate in (
                    input_dir / f"{stem}_gt.nii.gz",
                    input_dir / f"{stem}_gt.nii",
                )
                if candidate.is_file()
            ),
            None,
        )
        cases.append((volume, ground_truth))
    return cases


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one volume, or auto-pair every volume and ``*_gt`` in a directory."""
    if not args.input.is_dir():
        return _run_single(args)
    if args.ground_truth is not None:
        raise ValueError("--ground-truth cannot be used when input is a dataset directory")
    cases = _dataset_cases(args.input)
    if not cases:
        raise FileNotFoundError(f"No .nii or .nii.gz volumes found in {args.input}")
    from tqdm.auto import tqdm

    dataset_results: dict[str, str] = {}
    for input_path, ground_truth in tqdm(
        cases, desc="Dataset inference", unit="case"
    ):
        case_name = _nifti_stem(input_path)
        case_args = argparse.Namespace(
            **{
                **vars(args),
                "input": input_path,
                "ground_truth": ground_truth,
                "output": args.output / case_name,
                "metrics_output": None,
            }
        )
        result = _run_single(
            case_args,
            display_progress=False,
            display_perturbation_progress=True,
        )
        dataset_results[case_name] = result["output_directory"]
    summary = {
        "input_directory": str(args.input.resolve()),
        "output_directory": str(args.output.resolve()),
        "case_count": len(dataset_results),
        "cases": dataset_results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Dataset complete: {len(dataset_results)} cases")
    print(f"Result directory: {args.output}")
    print(f"Summary: {args.output / 'dataset_summary.json'}")
    return summary
