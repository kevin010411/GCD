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
        metric_curve_rows = []
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
                        scores = curve.get("scores", [])
                        prediction_dice = curve.get("prediction_dice", [])
                        prediction_iou = curve.get("prediction_iou", [])
                        ground_truth_dice = curve.get("ground_truth_dice", [])
                        ground_truth_iou = curve.get("ground_truth_iou", [])
                        curve_rows.append(
                            {
                                "method": result["method"],
                                "target_class": result["target_class"],
                                "operation": operation,
                                "variant": variant,
                                "step": index,
                                "fraction": fraction,
                                "target_probability": scores[index] if index < len(scores) else None,
                                "prediction_dice": prediction_dice[index] if index < len(prediction_dice) else None,
                                "prediction_iou": prediction_iou[index] if index < len(prediction_iou) else None,
                                "ground_truth_dice": ground_truth_dice[index] if index < len(ground_truth_dice) else None,
                                "ground_truth_iou": ground_truth_iou[index] if index < len(ground_truth_iou) else None,
                            }
                        )
                        for scorer_id, values in curve.get("curves", {}).items():
                            if index < len(values):
                                metric_curve_rows.append(
                                    {
                                        "method": result["method"],
                                        "target_class": result["target_class"],
                                        "operation": operation,
                                        "variant": variant,
                                        "scorer": scorer_id,
                                        "step": index,
                                        "fraction": fraction,
                                        "score": values[index],
                                        "auc": curve.get("aucs", {}).get(scorer_id),
                                    }
                                )
        _write_csv(summary_path, ["method", "target_class", "operation", "variant", "status", "answer_retention", "attribution", "auc", "reason"], summary_rows)
        _write_csv(curve_path, ["method", "target_class", "operation", "variant", "step", "fraction", "target_probability", "insertion_score", "deletion_score", "prediction_dice", "prediction_iou", "ground_truth_dice", "ground_truth_iou"], curve_rows)
        paths.extend((summary_path, curve_path))
        if metric_curve_rows:
            metric_curve_path = output_dir / "xai_metric_curves.csv"
            _write_csv(
                metric_curve_path,
                [
                    "method", "target_class", "operation", "variant",
                    "scorer", "step", "fraction", "score", "auc",
                ],
                metric_curve_rows,
            )
            paths.append(metric_curve_path)
        organ_rows = []
        for result in metrics["xai"].values():
            for organ in result.get("organs", []):
                organ_rows.append(
                    {
                        "method": result.get("method"),
                        "target_class": result.get("target_class"),
                        "objective": result.get("objective"),
                        **organ,
                    }
                )
        if organ_rows:
            organ_path = output_dir / "organ_occlusion_ranking.csv"
            _write_csv(
                organ_path,
                [
                    "method", "target_class", "objective", "rank", "organ_id",
                    "display_name", "source_labels", "baseline_score",
                    "occluded_score", "signed_delta", "voxel_count",
                ],
                organ_rows,
            )
            paths.append(organ_path)
    answer = metrics.get("xai_answer")
    if answer and answer.get("ranking"):
        answer_path = output_dir / "xai_answer.csv"
        _write_csv(
            answer_path,
            [
                "rank", "result_id", "method", "target_class",
                "insertion_auc", "deletion_auc", "final_score",
            ],
            answer["ranking"],
        )
        paths.append(answer_path)
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
            scorer_ids = sorted(
                {
                    scorer_id
                    for curve in completed.values()
                    for scorer_id in curve.get("curves", {})
                }
            )
            titles = {
                "target_probability": "Target-class probability",
                "prediction_dice": "Prediction Dice",
                "prediction_iou": "Prediction IoU",
                "ground_truth_dice": "Ground-truth Dice",
                "ground_truth_iou": "Ground-truth IoU",
            }
            bounded_scorers = set(titles)
            for metric_name in scorer_ids:
                title = titles.get(metric_name, metric_name.replace("_", " ").title())
                bounded = metric_name in bounded_scorers
                available = {
                    variant: curve
                    for variant, curve in completed.items()
                    if curve.get("curves", {}).get(metric_name)
                }
                if not available:
                    continue
                figure, axis = plt.subplots(figsize=(7, 4.5))
                for variant, curve in available.items():
                    label = variant
                    if metric_name == "target_probability":
                        label += f" (AUC={curve['aucs'][metric_name]:.4f})"
                    axis.plot(
                        curve["fractions"],
                        curve["curves"][metric_name],
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
