from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_csv_files(metrics: dict[str, Any], output_dir: Path) -> list[Path]:
    """Write analysis-friendly CSV tables from the complete metrics payload."""
    benchmark = metrics["benchmark"]
    summary = {
        "input": metrics["input"],
        "prediction": metrics["output"],
        "config": metrics["config"],
        "checkpoint": metrics["checkpoint"],
        "device": metrics["device"],
        "class_count": metrics["class_count"],
        "mean_latency_seconds": benchmark["mean_latency_seconds"],
        "volumes_per_second": benchmark["volumes_per_second"],
        "parameter_count": benchmark["parameter_count"],
    }
    if "segmentation" in metrics:
        summary["mean_dice"] = metrics["segmentation"]["mean_dice"]
        summary["mean_iou"] = metrics["segmentation"]["mean_iou"]

    paths = [output_dir / "summary.csv", output_dir / "benchmark_runs.csv", output_dir / "class_voxels.csv"]
    _write_csv(paths[0], list(summary), [summary])
    _write_csv(
        paths[1],
        ["run", "latency_seconds"],
        [
            {"run": index, "latency_seconds": latency}
            for index, latency in enumerate(benchmark["latency_seconds"], start=1)
        ],
    )
    _write_csv(
        paths[2],
        ["class_id", "predicted_voxels"],
        [
            {"class_id": class_id, "predicted_voxels": count}
            for class_id, count in metrics["class_voxels"].items()
        ],
    )

    if "segmentation" in metrics:
        path = output_dir / "segmentation_metrics.csv"
        rows = [
            {"class_id": class_id, **values}
            for class_id, values in metrics["segmentation"]["per_class"].items()
        ]
        _write_csv(path, ["class_id", "dice", "iou", "predicted_voxels", "target_voxels"], rows)
        paths.append(path)

    if metrics.get("xai"):
        summary_path = output_dir / "xai_summary.csv"
        curve_path = output_dir / "xai_curves.csv"
        summary_rows = []
        curve_rows = []
        for result_id, result in metrics["xai"].items():
            if "perturbations" not in result:
                summary_rows.append(
                    {
                        "method": result_id,
                        "target_class": result["target_class"],
                        "operation": "insertion/deletion",
                        "variant": "legacy",
                        "status": "completed",
                        "attribution": result["attribution"],
                    }
                )
                fractions = result.get("fractions", [])
                insertion = result.get("insertion_scores", [])
                deletion = result.get("deletion_scores", [])
                for index, fraction in enumerate(fractions):
                    curve_rows.append(
                        {
                            "method": result_id,
                            "target_class": result["target_class"],
                            "operation": "insertion/deletion",
                            "variant": "legacy",
                            "step": index,
                            "fraction": fraction,
                            "target_probability": insertion[index] if index < len(insertion) else None,
                            "insertion_score": insertion[index] if index < len(insertion) else None,
                            "deletion_score": deletion[index] if index < len(deletion) else None,
                        }
                    )
                continue
            for operation, variants in result.get("perturbations", {}).items():
                for variant, curve in variants.items():
                    summary_rows.append(
                        {
                            "method": result["method"],
                            "target_class": result["target_class"],
                            "operation": operation,
                            "variant": variant,
                            "status": curve.get("status"),
                            "answer_retention": curve.get("answer_retention"),
                            "attribution": result["attribution"],
                            "auc": curve.get("auc"),
                            "reason": curve.get("reason"),
                        }
                    )
                    for index, fraction in enumerate(curve.get("fractions", [])):
                        curve_rows.append(
                            {
                                "method": result["method"],
                                "target_class": result["target_class"],
                                "operation": operation,
                                "variant": variant,
                                "step": index,
                                "fraction": fraction,
                                "target_probability": curve["scores"][index],
                                "prediction_dice": curve["prediction_dice"][index],
                                "prediction_iou": curve["prediction_iou"][index],
                                "ground_truth_dice": curve["ground_truth_dice"][index] if curve["ground_truth_dice"] else None,
                                "ground_truth_iou": curve["ground_truth_iou"][index] if curve["ground_truth_iou"] else None,
                            }
                        )
        _write_csv(summary_path, ["method", "target_class", "operation", "variant", "status", "answer_retention", "attribution", "auc", "reason"], summary_rows)
        _write_csv(curve_path, ["method", "target_class", "operation", "variant", "step", "fraction", "target_probability", "insertion_score", "deletion_score", "prediction_dice", "prediction_iou", "ground_truth_dice", "ground_truth_iou"], curve_rows)
        paths.extend((summary_path, curve_path))
    return paths


def export_faithfulness_plots(metrics: dict[str, Any], output_dir: Path) -> list[Path]:
    """Create one insertion/deletion line plot for each XAI result with a curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for result_id, result in metrics.get("xai", {}).items():
        if "perturbations" not in result:
            fractions = result.get("fractions", [])
            insertion = result.get("insertion_scores", [])
            deletion = result.get("deletion_scores", [])
            if not fractions or (not insertion and not deletion):
                continue
            figure, axis = plt.subplots(figsize=(7, 4.5))
            if insertion:
                axis.plot(fractions, insertion, marker="o", label="Insertion")
            if deletion:
                axis.plot(fractions, deletion, marker="o", label="Deletion")
            axis.set(
                title=f"{result_id} faithfulness",
                xlabel="Perturbed fraction",
                ylabel="Target probability",
            )
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            path = output_dir / f"{result_id}_insertion_deletion.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(path)
            continue
        for operation, variants in result.get("perturbations", {}).items():
            completed = {
                name: curve
                for name, curve in variants.items()
                if curve.get("status") == "completed"
            }
            if not completed:
                continue
            operation_name = operation.rsplit("_", 1)[0]
            plot_dir = (
                output_dir
                / "plots"
                / result["method"]
                / f"class_{result['target_class']}"
                / operation
            )
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_specs = (
                ("target_probability", "scores", "Target-class probability", False),
                ("prediction_dice", "prediction_dice", "Prediction Dice", True),
                ("prediction_iou", "prediction_iou", "Prediction IoU", True),
                ("ground_truth_dice", "ground_truth_dice", "Ground-truth Dice", True),
                ("ground_truth_iou", "ground_truth_iou", "Ground-truth IoU", True),
            )
            for metric_name, value_key, title, bounded in plot_specs:
                available = {
                    variant: curve
                    for variant, curve in completed.items()
                    if curve[value_key]
                }
                if not available:
                    continue
                figure, axis = plt.subplots(figsize=(7, 4.5))
                for variant, curve in available.items():
                    label = variant
                    if metric_name == "target_probability":
                        label += f" (AUC={curve['auc']:.4f})"
                    axis.plot(
                        curve["fractions"],
                        curve[value_key],
                        marker="o",
                        label=label,
                    )
                axis.set(
                    title=(
                        f"{result['method']} / class {result['target_class']} / "
                        f"{operation} / {title}"
                    ),
                    xlabel="Perturbed eligible fraction",
                    ylabel=title,
                )
                if bounded:
                    axis.set_ylim(-0.02, 1.02)
                axis.grid(True, alpha=0.3)
                axis.legend(fontsize="small")
                figure.tight_layout()
                path = plot_dir / (
                    f"{operation_name}_{result['target_class']}_{metric_name}.png"
                )
                figure.savefig(path, dpi=160)
                plt.close(figure)
                paths.append(path)
    return paths
