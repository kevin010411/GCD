from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VolumeLoadConfig:
    spacing: tuple[float, float, float]
    permute: tuple[int, int, int]
    default_layer: str


@dataclass(frozen=True)
class LoadedVolume:
    file_name: str
    origin_img: object
    origin_meta: dict[str, object]
    origin_shape: tuple[int, ...]
    img0: object
    display_volume: object
    display_metadata: dict[str, object]
    display_spacing: tuple[float, float, float]
    default_layers: dict[str, int]
    messages: list[str]


def safe_affine(
    origin_meta: dict[str, Any] | None,
    spacing: tuple[float, float, float],
):
    import numpy as np

    metadata = origin_meta or {}
    affine = metadata.get("affine")
    if affine is not None:
        return np.array(affine, dtype=np.float32, copy=True)

    pixdim = metadata.get("pixdim")
    extracted_spacing = None
    if pixdim is not None and len(pixdim) >= 4:
        try:
            extracted_spacing = tuple(map(float, pixdim[1:4]))
        except Exception:
            extracted_spacing = None
    if extracted_spacing is None:
        extracted_spacing = tuple(float(value) for value in spacing)
    return np.diag([*extracted_spacing, 1.0]).astype("float32")


def extract_affine(image, fallback_affine):
    import numpy as np

    meta = getattr(image, "meta", None)
    if meta is not None:
        affine = meta.get("affine")
        if affine is not None:
            return np.array(affine, dtype=np.float32, copy=True)
    return np.array(fallback_affine, dtype=np.float32, copy=True)


def shift_affine_for_padding(affine, offsets: list[int] | tuple[int, int, int]):
    import numpy as np

    shifted = np.array(affine, dtype=np.float32, copy=True)
    offset_vector = np.array(offsets, dtype=np.float32)
    shifted[:3, 3] -= shifted[:3, :3] @ offset_vector
    return shifted


def build_display_metadata(affine, permute: tuple[int, int, int]) -> dict[str, object]:
    import numpy as np

    axis_order = list(permute)
    display_affine = np.eye(4, dtype=np.float32)
    display_affine[:3, :3] = affine[:3, :3][:, axis_order]
    display_affine[:3, 3] = affine[:3, 3]

    display_vectors = display_affine[:3, :3]
    display_spacing = np.linalg.norm(display_vectors, axis=0)
    safe_display_spacing = np.where(display_spacing > 0, display_spacing, 1.0)
    display_direction = display_vectors / safe_display_spacing

    vtk_axis_order = [2, 1, 0]
    vtk_vectors = display_vectors[:, vtk_axis_order]
    vtk_spacing = np.linalg.norm(vtk_vectors, axis=0)
    safe_vtk_spacing = np.where(vtk_spacing > 0, vtk_spacing, 1.0)
    vtk_direction = vtk_vectors / safe_vtk_spacing

    return {
        "affine": display_affine,
        "origin": tuple(float(v) for v in display_affine[:3, 3]),
        "spacing": tuple(float(v) for v in safe_display_spacing),
        "direction": tuple(
            tuple(float(v) for v in row) for row in display_direction.T
        ),
        "vtk_origin": tuple(float(v) for v in display_affine[:3, 3]),
        "vtk_spacing": tuple(float(v) for v in safe_vtk_spacing),
        "vtk_direction": tuple(tuple(float(v) for v in row) for row in vtk_direction.T),
    }


class VolumeLoadingService:
    def load(self, file_name: str, config: VolumeLoadConfig) -> LoadedVolume:
        import monai.transforms as mt
        import torch

        origin_img, origin_meta = mt.LoadImage(image_only=False)(file_name)
        img0 = mt.EnsureChannelFirst()(origin_img, origin_meta)
        try:
            origin_meta = dict(img0.meta)
        except Exception:
            origin_meta = dict(origin_meta or {})
        origin_shape = tuple(img0.shape)
        source_affine = safe_affine(origin_meta, config.spacing)
        raw_affine = extract_affine(img0, source_affine)
        display_metadata = build_display_metadata(raw_affine, config.permute)
        display_metadata.update(
            {
                "source_affine": source_affine,
                "source_shape": tuple(int(v) for v in origin_shape[1:]),
                "display_permute": tuple(int(v) for v in config.permute),
            }
        )
        display_volume = img0[0].permute(*config.permute).to(torch.float32)
        display_spacing = tuple(float(v) for v in display_metadata["spacing"])
        return LoadedVolume(
            file_name=file_name,
            origin_img=origin_img,
            origin_meta=origin_meta,
            origin_shape=origin_shape,
            img0=img0,
            display_volume=display_volume,
            display_metadata=display_metadata,
            display_spacing=display_spacing,
            default_layers={config.default_layer: 1},
            messages=[],
        )
