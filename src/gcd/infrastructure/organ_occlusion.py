from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

from ..domain import (
    OrganIntervention,
    OrganMaskRecord,
    OrganOcclusionResult,
    OrganOcclusionSpec,
)


class LabelMaskView:
    """Lazy boolean view over a shared TotalSegmentator multilabel array."""

    def __init__(self, labelmap: np.ndarray, label_ids: Iterable[int]) -> None:
        self.labelmap = labelmap
        self.label_ids = tuple(int(value) for value in label_ids)
        self.shape = labelmap.shape

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        result = np.isin(self.labelmap, self.label_ids)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = np.array(result, copy=True)
        return result


CURATED_ORGANS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "lung_left",
        "左肺 / Left lung",
        ("lung_upper_lobe_left", "lung_lower_lobe_left"),
        "#38BDF8",
    ),
    ("heart", "心臟 / Heart", ("heart",), "#F43F5E"),
    (
        "lung_right",
        "右肺 / Right lung",
        ("lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"),
        "#0EA5E9",
    ),
    ("heart_myocardium", "心肌 / Myocardium", ("heart_myocardium",), "#EF4444"),
    ("heart_atrium_left", "左心房 / Left atrium", ("heart_atrium_left",), "#FB7185"),
    ("heart_ventricle_left", "左心室 / Left ventricle", ("heart_ventricle_left",), "#E11D48"),
    ("heart_atrium_right", "右心房 / Right atrium", ("heart_atrium_right",), "#F97316"),
    ("heart_ventricle_right", "右心室 / Right ventricle", ("heart_ventricle_right",), "#EA580C"),
    ("aorta", "主動脈 / Aorta", ("aorta",), "#FACC15"),
    ("pulmonary_artery", "肺動脈 / Pulmonary artery", ("pulmonary_artery",), "#A78BFA"),
    (
        "thoracic_spine",
        "胸椎 / Thoracic spine",
        tuple(f"vertebrae_T{index}" for index in range(1, 13)),
        "#E5E7EB",
    ),
)


def _as_array(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def validate_mask_geometry(
    image: np.ndarray,
    image_affine,
    mask: np.ndarray,
    mask_affine,
    *,
    atol: float = 1e-3,
) -> None:
    if tuple(image.shape) != tuple(mask.shape):
        raise ValueError(
            f"Organ mask shape {tuple(mask.shape)} does not match CT shape {tuple(image.shape)}."
        )
    if image_affine is not None and mask_affine is not None:
        if not np.allclose(_as_array(image_affine), _as_array(mask_affine), atol=atol):
            raise ValueError("Organ mask affine does not match the source CT affine.")


def _spacing_from_affine(affine) -> tuple[float, float, float]:
    matrix = _as_array(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        return (1.0, 1.0, 1.0)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    return tuple(float(value if value > 0 else 1.0) for value in spacing)


def _radius_voxels(mm: float, spacing: tuple[float, float, float]) -> tuple[float, ...]:
    return tuple(max(0.0, float(mm) / max(float(axis), 1e-6)) for axis in spacing)


def _feather_alpha(
    mask: np.ndarray, feather_mm: float, spacing: tuple[float, float, float]
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if feather_mm <= 0:
        return binary.astype(np.float32)
    layers = max(1, int(np.ceil(float(feather_mm) / min(spacing))))
    alpha = np.zeros(binary.shape, dtype=np.float32)
    current = binary
    for level in range(1, layers + 1):
        eroded = ndimage.binary_erosion(current)
        alpha[current & ~eroded] = float(level) / float(layers)
        current = eroded
        if not np.any(current):
            break
    alpha[current] = 1.0
    return alpha


def _local_mean_candidate(
    image: np.ndarray,
    mask: np.ndarray,
    shell_mm: float,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    iterations = max(1, int(np.ceil(float(shell_mm) / min(spacing))))
    shell = ndimage.binary_dilation(mask, iterations=iterations) & ~mask
    values = image[shell & np.isfinite(image)]
    if values.size == 0:
        values = image[~mask & np.isfinite(image)]
    fill = float(np.mean(values)) if values.size else 0.0
    return np.full_like(image, fill, dtype=np.float32)


def _mask_region(
    mask: np.ndarray,
    margin_mm: float,
    spacing: tuple[float, float, float],
) -> tuple[slice, slice, slice]:
    coordinates = np.where(mask)
    if not coordinates[0].size:
        return (slice(0, 0), slice(0, 0), slice(0, 0))
    margins = [int(np.ceil(float(margin_mm) / max(axis, 1e-6))) for axis in spacing]
    return tuple(
        slice(
            max(0, int(axis_values.min()) - margins[index]),
            min(mask.shape[index], int(axis_values.max()) + margins[index] + 1),
        )
        for index, axis_values in enumerate(coordinates)
    )


def apply_organ_occlusion(
    image,
    organ_masks: dict[str, OrganMaskRecord],
    spec: OrganOcclusionSpec,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    source = _as_array(image, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("Organ occlusion expects a 3D CT volume.")
    enabled = spec.enabled()
    if not enabled:
        raise ValueError("Select at least one organ before running Organ Occlusion.")

    result = np.array(source, copy=True)
    modified_mask = np.zeros(source.shape, dtype=bool)
    for intervention in enabled:
        if intervention.organ_id not in organ_masks:
            raise KeyError(f"Unknown organ mask: {intervention.organ_id}")
        if not -1000.0 <= float(intervention.fill_hu) <= 1000.0:
            raise ValueError("Fixed HU must be between -1000 and 1000.")
        mask = _as_array(organ_masks[intervention.organ_id].mask, dtype=bool)
        if mask.shape != source.shape:
            raise ValueError(f"Mask shape mismatch for organ {intervention.organ_id}.")
        if not np.any(mask):
            continue

        margin_mm = max(
            float(spec.feather_mm),
            float(spec.local_mean_shell_mm),
            float(spec.blur_sigma_mm) * 3.0,
        )
        region = _mask_region(mask, margin_mm, spacing)
        source_region = source[region]
        mask_region = mask[region]
        if intervention.mode == "fixed_hu":
            candidate = np.full_like(
                source_region, float(intervention.fill_hu), dtype=np.float32
            )
        elif intervention.mode == "gaussian_blur":
            sigma = _radius_voxels(spec.blur_sigma_mm, spacing)
            candidate = ndimage.gaussian_filter(source_region, sigma=sigma).astype(np.float32)
        elif intervention.mode == "local_mean":
            candidate = _local_mean_candidate(
                source_region, mask_region, spec.local_mean_shell_mm, spacing
            )
        else:
            raise ValueError(f"Unsupported organ occlusion mode: {intervention.mode}")

        alpha = _feather_alpha(mask_region, spec.feather_mm, spacing)
        result_region = result[region]
        result[region] = result_region * (1.0 - alpha) + candidate * alpha
        modified_mask |= mask
    return result.astype(np.float32, copy=False), modified_mask


def prediction_metrics(
    original_prediction,
    perturbed_prediction,
    target_class: int,
    modified_mask,
    *,
    valid_mask=None,
    ground_truth=None,
) -> tuple[np.ndarray, dict[str, float]]:
    original = _as_array(original_prediction)
    perturbed = _as_array(perturbed_prediction)
    if original.shape != perturbed.shape:
        raise ValueError("Original and perturbed predictions must have the same shape.")
    original_target = original == int(target_class)
    perturbed_target = perturbed == int(target_class)
    intersection = np.logical_and(original_target, perturbed_target).sum()
    union = np.logical_or(original_target, perturbed_target).sum()
    denom = original_target.sum() + perturbed_target.sum()
    dice = 1.0 if denom == 0 else (2.0 * float(intersection)) / float(denom)
    iou = 1.0 if union == 0 else float(intersection) / float(union)
    original_volume = int(original_target.sum())
    perturbed_volume = int(perturbed_target.sum())
    volume_change = (
        0.0
        if original_volume == 0
        else (float(perturbed_volume - original_volume) / float(original_volume))
    )
    changed = _as_array(modified_mask, dtype=bool)
    denominator_mask = (
        _as_array(valid_mask, dtype=bool)
        if valid_mask is not None
        else np.isfinite(original)
    )
    metrics = {
        "stability_dice": float(dice),
        "stability_iou": float(iou),
        "target_volume_change_fraction": float(volume_change),
        "modified_voxels": float(changed.sum()),
        "modified_fraction": float(changed.sum() / max(1, denominator_mask.sum())),
    }
    if ground_truth is not None:
        truth = _as_array(ground_truth) == int(target_class)

        def _dice(mask: np.ndarray) -> float:
            overlap = np.logical_and(mask, truth).sum()
            total = mask.sum() + truth.sum()
            return 1.0 if total == 0 else (2.0 * float(overlap)) / float(total)

        def _iou(mask: np.ndarray) -> float:
            overlap = np.logical_and(mask, truth).sum()
            total = np.logical_or(mask, truth).sum()
            return 1.0 if total == 0 else float(overlap) / float(total)

        original_dice = _dice(original_target)
        perturbed_dice = _dice(perturbed_target)
        original_iou = _iou(original_target)
        perturbed_iou = _iou(perturbed_target)
        metrics.update(
            {
                "ground_truth_dice_before": original_dice,
                "ground_truth_dice_after": perturbed_dice,
                "ground_truth_dice_delta": perturbed_dice - original_dice,
                "ground_truth_iou_before": original_iou,
                "ground_truth_iou_after": perturbed_iou,
                "ground_truth_iou_delta": perturbed_iou - original_iou,
            }
        )

    difference = np.zeros(original.shape, dtype=np.uint8)
    difference[np.logical_and(original_target, perturbed_target)] = 1
    difference[np.logical_and(original_target, ~perturbed_target)] = 2
    difference[np.logical_and(~original_target, perturbed_target)] = 3
    return difference, metrics


class TotalSegmentatorOrganService:
    def __init__(self, cache_root: str | Path = "output/organ_masks") -> None:
        self.cache_root = Path(cache_root)
        self.last_run_metadata: dict[str, object] = {}

    @staticmethod
    def source_hash(path: str | Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def cache_key(self, path: str | Path, version: str, task: str = "total") -> str:
        return hashlib.sha256(
            f"{self.source_hash(path)}|{version}|{task}".encode("utf-8")
        ).hexdigest()[:24]

    def run(
        self,
        source_path: str | Path,
        *,
        force: bool = False,
        device: str = "gpu",
        task: str = "total",
        merge_organs: bool = True,
    ) -> list[OrganMaskRecord]:
        import nibabel as nib
        import totalsegmentator
        from totalsegmentator.registry import get_task_classes

        version = str(getattr(totalsegmentator, "__version__", "unknown"))
        image = nib.load(str(source_path))
        tasks = [task]
        if task == "total" and self._has_totalsegmentator_license():
            tasks.append("heartchambers_highres")
        records_by_id: dict[str, OrganMaskRecord] = {}
        task_metadata: list[dict[str, object]] = []
        for current_task in tasks:
            key = self.cache_key(source_path, version, current_task)
            label_path = self.cache_root / key / f"{current_task}.nii.gz"
            cache_hit = label_path.exists() and not force
            label_path = self._ensure_task_output(
                source_path,
                version=version,
                task=current_task,
                force=force,
                device=device,
            )
            labels = nib.load(str(label_path))
            validate_mask_geometry(
                np.asarray(image.dataobj),
                image.affine,
                np.asarray(labels.dataobj),
                labels.affine,
            )
            for record in self._records_from_labelmap(
                np.asarray(labels.dataobj), labels.affine, get_task_classes(current_task),
                merge_organs=merge_organs,
            ):
                records_by_id[record.id] = record
            task_metadata.append(
                {
                    "task": current_task,
                    "cache_key": key,
                    "cache_hit": cache_hit,
                    "label_path": str(label_path.resolve()),
                }
            )
        self.last_run_metadata = {
            "version": version,
            "cache_root": str(self.cache_root.resolve()),
            "tasks": task_metadata,
            "merge_organs": bool(merge_organs),
        }
        return list(records_by_id.values())

    @staticmethod
    def _has_totalsegmentator_license() -> bool:
        try:
            from totalsegmentator.config import get_license_number

            return bool(get_license_number())
        except Exception:
            return False

    def _ensure_task_output(
        self,
        source_path: str | Path,
        *,
        version: str,
        task: str,
        force: bool,
        device: str,
    ) -> Path:
        import nibabel as nib
        from totalsegmentator.python_api import totalsegmentator as run_total

        key = self.cache_key(source_path, version, task)
        cache_dir = self.cache_root / key
        label_path = cache_dir / f"{task}.nii.gz"
        manifest_path = cache_dir / "manifest.json"
        if not force and label_path.exists():
            return label_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        result = run_total(
            input=Path(source_path),
            output=label_path,
            ml=True,
            task=task,
            device=device,
        )
        if not label_path.exists() and result is not None:
            nib.save(result, label_path)
        if not label_path.exists():
            raise RuntimeError(f"TotalSegmentator task '{task}' produced no volume.")
        manifest_path.write_text(
            json.dumps(
                {
                    "source": str(source_path),
                    "source_hash": self.source_hash(source_path),
                    "totalsegmentator_version": version,
                    "task": task,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return label_path

    @staticmethod
    def _records_from_labelmap(
        labelmap: np.ndarray,
        affine,
        class_map: dict[int, str],
        *,
        merge_organs: bool = True,
    ) -> list[OrganMaskRecord]:
        inverse = {str(name): int(index) for index, name in class_map.items()}
        present_ids = {int(value) for value in np.unique(labelmap)}
        records: list[OrganMaskRecord] = []
        consumed: set[str] = set()
        for organ_id, name, source_labels, color in (
            CURATED_ORGANS if merge_organs else ()
        ):
            ids = [inverse[label] for label in source_labels if label in inverse]
            ids = [value for value in ids if value in present_ids]
            if not ids:
                continue
            consumed.update(source_labels)
            records.append(
                OrganMaskRecord(
                    id=organ_id,
                    display_name=name,
                    source_labels=source_labels,
                    mask=LabelMaskView(labelmap, ids),
                    affine=np.array(affine, copy=True),
                    color=color,
                    group="curated",
                )
            )
        for index, label in sorted(class_map.items()):
            label = str(label)
            if label in consumed:
                continue
            if int(index) not in present_ids:
                continue
            records.append(
                OrganMaskRecord(
                    id=label,
                    display_name=label.replace("_", " ").title(),
                    source_labels=(label,),
                    mask=LabelMaskView(labelmap, (int(index),)),
                    affine=np.array(affine, copy=True),
                    color="#94A3B8",
                    group="advanced",
                )
            )
        if not records:
            raise ValueError("TotalSegmentator returned no non-empty organ masks.")
        return records


def result_metadata(result: OrganOcclusionResult) -> dict[str, object]:
    return {
        **dict(result.metadata),
        "metrics": dict(result.metrics),
        "interventions": [asdict(item) for item in result.spec.interventions],
        "feather_mm": result.spec.feather_mm,
        "blur_sigma_mm": result.spec.blur_sigma_mm,
        "local_mean_shell_mm": result.spec.local_mean_shell_mm,
    }
