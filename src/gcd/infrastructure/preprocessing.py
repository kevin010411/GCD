"""GCD-wide config-driven preprocessing for file and in-memory volumes."""
from pathlib import Path


def preprocess_tensor(image, cfg, *, label=False, roi=None):
    import torch
    import monai.transforms as mt

    settings = cfg.preprocessing
    if settings.get("force_3d", False) and image.ndim == 3:
        image = image.unsqueeze(-1)
    steps = settings.get("steps")
    if steps is None:
        steps = [dict(type="Spacing", mode="bilinear", label_mode="nearest"),
                 dict(type="ScaleIntensityRange", clip=True)]
    transforms = []
    for item in steps:
        params = dict(item)
        name = params.pop("type")
        if name == "Spacing":
            label_mode = params.pop("label_mode", "nearest")
            if label:
                params["mode"] = label_mode
            params.setdefault("pixdim", tuple(settings.spacing))
        elif name == "SpatialPad":
            if params.get("spatial_size") == "window":
                params["spatial_size"] = tuple(roi or cfg.inference.roi_size)
            elif params.get("spatial_size") == "legacy_window":
                raise ValueError("legacy_window 已移除；請使用 spatial_size='window'。")
        elif name in {"NormalizeIntensity", "ScaleIntensityRange"}:
            if label:
                continue
            if name == "ScaleIntensityRange":
                params.setdefault("a_min", float(settings.intensity_input_range[0]))
                params.setdefault("a_max", float(settings.intensity_input_range[1]))
                params.setdefault("b_min", float(settings.intensity_output_range[0]))
                params.setdefault("b_max", float(settings.intensity_output_range[1]))
        elif name == "EnsureType":
            params["dtype"] = getattr(torch, params.get("dtype", "float32"))
        elif name != "Orientation":
            raise ValueError(f"Unsupported preprocessing transform: {name}")
        transforms.append(getattr(mt, name)(**params))
    result = mt.Compose(transforms)(image)
    # CAM windows must be complete; match the benchmark's final ROI padding
    # even when a legacy/configured SpatialPad used a smaller spatial_size.
    window = tuple(roi or cfg.inference.roi_size)
    if any(length < minimum for length, minimum in zip(result.shape[-3:], window)):
        inference = cfg.inference
        mode = inference.get("padding_mode", "constant")
        extra = {"constant_values": inference.get("cval", 0.0)} if mode == "constant" else {}
        result = mt.SpatialPad(window, method="symmetric", mode=mode, **extra)(result)
    return result


def load_image(path: Path, cfg, *, label=False, roi=None):
    import monai.transforms as mt

    image = mt.LoadImage(image_only=True)(str(path))
    image = mt.EnsureChannelFirst(channel_dim=cfg.preprocessing.get("channel_dim"))(image)
    return preprocess_tensor(image, cfg, label=label, roi=roi)


def preprocess_array(array, affine, cfg, *, label=False):
    import torch
    from monai.data import MetaTensor

    image = MetaTensor(torch.as_tensor(array, dtype=torch.float32).unsqueeze(0),
                       affine=torch.as_tensor(affine, dtype=torch.float64))
    return preprocess_tensor(image, cfg, label=label)
