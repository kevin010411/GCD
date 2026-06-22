from __future__ import annotations

from dataclasses import dataclass

from .volume_loading import (
    build_display_metadata,
    extract_affine,
    safe_affine,
    shift_affine_for_padding,
)


@dataclass(frozen=True)
class ModelInputPreprocessConfig:
    size: int
    stride: int
    spacing: tuple[float, float, float]
    permute: tuple[int, int, int]
    intensity_input_range: tuple[float, float] = (-42.0, 423.0)
    intensity_output_range: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class ModelInputVolume:
    img0: object
    img1: object
    origin_meta: dict[str, object]
    origin_shape: tuple[int, ...]
    model_input_metadata: dict[str, object]
    img1_spacing: tuple[float, float, float]
    messages: list[str]


class ModelInputPreprocessor:
    def preprocess(
        self,
        *,
        img0: object | None,
        origin_img: object | None,
        origin_meta: dict[str, object] | None,
        origin_shape: tuple[int, ...] | None,
        config: ModelInputPreprocessConfig,
    ) -> ModelInputVolume:
        import monai.transforms as mt
        import torch

        metadata = dict(origin_meta or {})
        if img0 is None:
            if origin_img is None:
                raise ValueError("缺少原始資料，無法產生模型輸入。")
            img0 = mt.EnsureChannelFirst()(origin_img, metadata)
            try:
                metadata = dict(img0.meta)
            except Exception:
                metadata = dict(metadata or {})
            origin_shape = tuple(img0.shape)
        elif origin_shape is None:
            origin_shape = tuple(getattr(img0, "shape", ()))

        messages: list[str] = []
        img1 = mt.Spacing(mode="bilinear", pixdim=config.spacing)(img0)
        width, depth = config.size + config.stride, config.size
        img1 = mt.SpatialPad(
            spatial_size=(width, width, depth), mode="constant", value=0
        )(img1)
        source_affine = safe_affine(metadata, config.spacing)
        img1_affine = extract_affine(img1, source_affine)

        shape = list(img1.shape)
        slices = [slice(None), slice(None), slice(None), slice(None)]
        pad_offsets = [0, 0, 0]
        if shape[1] < width:
            x = (width - shape[1]) // 2
            slices[1] = slice(x, x + shape[1])
            shape[1] = width
            pad_offsets[0] = x
        if shape[2] < width:
            x = (width - shape[2]) // 2
            slices[2] = slice(x, x + shape[2])
            shape[2] = width
            pad_offsets[1] = x
        if shape[3] < depth:
            x = (depth - shape[3]) // 2
            slices[3] = slice(x, x + shape[3])
            shape[3] = depth
            pad_offsets[2] = x
        if any(current.start is not None for current in slices):
            image = torch.zeros(shape)
            image[tuple(slices)] = img1
            img1 = image
            messages.append("info: image is zero padded")
            img1_affine = shift_affine_for_padding(img1_affine, pad_offsets)

        a_min, a_max = config.intensity_input_range
        b_min, b_max = config.intensity_output_range
        img1 = mt.ScaleIntensityRange(
            a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
        )(img1)
        img1_spacing = (
            config.spacing[config.permute[0]],
            config.spacing[config.permute[1]],
            config.spacing[config.permute[2]],
        )
        model_input_metadata = build_display_metadata(img1_affine, config.permute)
        model_input_metadata.update(
            {
                "source_affine": source_affine,
                "source_shape": tuple(int(v) for v in tuple(origin_shape or ())[1:]),
                "display_permute": tuple(int(v) for v in config.permute),
            }
        )
        return ModelInputVolume(
            img0=img0,
            img1=img1,
            origin_meta=metadata,
            origin_shape=tuple(origin_shape or ()),
            model_input_metadata=model_input_metadata,
            img1_spacing=tuple(float(v) for v in img1_spacing),
            messages=messages,
        )
