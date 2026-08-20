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
        for method, result in metrics["xai"].items():
            summary_rows.append(
                {
                    "method": method,
                    "target_class": result["target_class"],
                    "attribution": result["attribution"],
                    "insertion_auc": result.get("insertion_auc"),
                    "deletion_auc": result.get("deletion_auc"),
                }
            )
            fractions = result.get("fractions", [])
            insertion = result.get("insertion_scores", [])
            deletion = result.get("deletion_scores", [])
            for index, fraction in enumerate(fractions):
                curve_rows.append(
                    {
                        "method": method,
                        "step": index,
                        "fraction": fraction,
                        "insertion_score": insertion[index] if index < len(insertion) else None,
                        "deletion_score": deletion[index] if index < len(deletion) else None,
                    }
                )
        _write_csv(summary_path, ["method", "target_class", "attribution", "insertion_auc", "deletion_auc"], summary_rows)
        _write_csv(curve_path, ["method", "step", "fraction", "insertion_score", "deletion_score"], curve_rows)
        paths.extend((summary_path, curve_path))
    return paths


def export_faithfulness_plots(metrics: dict[str, Any], output_dir: Path) -> list[Path]:
    """Create one insertion/deletion line plot for each XAI result with a curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for method, result in metrics.get("xai", {}).items():
        fractions = result.get("fractions", [])
        insertion = result.get("insertion_scores", [])
        deletion = result.get("deletion_scores", [])
        if not fractions or (not insertion and not deletion):
            continue
        figure, axis = plt.subplots(figsize=(7, 4.5))
        if insertion:
            axis.plot(fractions, insertion, marker="o", label=f"Insertion (AUC={result['insertion_auc']:.4f})")
        if deletion:
            axis.plot(fractions, deletion, marker="o", label=f"Deletion (AUC={result['deletion_auc']:.4f})")
        axis.set(title=f"{method} faithfulness", xlabel="Perturbed fraction", ylabel="Target probability")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"{method}_insertion_deletion.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths
