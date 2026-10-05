"""Qt-independent NIfTI geometry and configured sliding-window evaluation."""
from pathlib import Path

import numpy as np

from ..preprocessing import load_image
from .methods.benchmark_cam import normalize
from .runtime.model_runtime_loader import ModelRuntimeLoader


def nifti_volume(path):
    import nibabel as nib
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected one 3-D volume, got {data.shape}: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite NIfTI values: {path}")
    affine = np.asarray(image.affine)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError(f"Invalid NIfTI affine: {path}")
    return data, affine


def require_same_grid(shape, affine, other_shape, other_affine, description):
    if tuple(shape) != tuple(other_shape) or not np.allclose(
        affine, other_affine, atol=1e-4, rtol=0
    ):
        raise ValueError(f"{description}: shape/affine differ; align the physical grids explicitly")


def load_case(image_path, label_path, cfg, roi=None):
    raw, raw_affine = nifti_volume(image_path)
    truth, truth_affine = nifti_volume(label_path)
    require_same_grid(raw.shape, raw_affine, truth.shape, truth_affine, "Image/label")
    if np.any(truth < 0) or not np.allclose(truth, np.round(truth), atol=1e-5, rtol=0):
        raise ValueError("Label NIfTI must contain nonnegative integer class IDs")
    image = load_image(Path(image_path), cfg, roi=roi)
    label = load_image(Path(label_path), cfg, label=True, roi=roi)
    require_same_grid(image.shape[-3:], np.asarray(image.affine),
                      label.shape[-3:], np.asarray(label.affine), "Preprocessed image/label")
    return image, label


def load_heatmap(path, source_path, image_meta, space="original"):
    """Resample attribution by world coordinates, NEVER CT intensity transforms."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    data, affine = nifti_volume(path)
    if np.any(data < 0):
        raise ValueError("Heatmap must be nonnegative; signed attributions need an explicit protocol")
    target = (tuple(image_meta.shape[-3:]), np.asarray(image_meta.affine))
    source, source_affine = nifti_volume(source_path)
    if space == "original":
        require_same_grid(source.shape, source_affine, data.shape, affine, "Original image/heatmap")
    elif space == "model":
        require_same_grid(*target, data.shape, affine, "Model image/heatmap")
    elif space != "world":
        raise ValueError("heatmap space must be original, model, or world")
    if space == "model":
        result = data.astype(np.float32)
    else:
        # Detect completely unrelated world-space data instead of scoring a zero map.
        corners = np.array(np.meshgrid(*[(0, n - 1) for n in data.shape], indexing="ij")).reshape(3, -1)
        world = affine[:3, :3] @ corners + affine[:3, 3:4]
        src_corners = np.array(np.meshgrid(*[(0, n - 1) for n in source.shape], indexing="ij")).reshape(3, -1)
        src_world = source_affine[:3, :3] @ src_corners + source_affine[:3, 3:4]
        if np.any(world.max(1) < src_world.min(1)) or np.any(src_world.max(1) < world.min(1)):
            raise ValueError("Heatmap and image do not overlap in world space")
        result = np.asarray(resample_from_to(
            nib.Nifti1Image(data.astype(np.float32), affine), target,
            order=1, mode="constant", cval=0,
        ).dataobj, dtype=np.float32)
    return normalize(result)


def prediction(logits, class_id: int, connected_components: bool):
    prediction = logits.argmax(dim=1)[0].detach().cpu().numpy()
    if connected_components:
        from scipy import ndimage
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        prediction = prediction.copy()
        for label in np.unique(prediction):
            if label == 0:
                continue
            mask = prediction == label
            components, count = ndimage.label(mask, structure)
            if count > 1:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                prediction[mask & (components != int(sizes.argmax()))] = 0
    probability = logits.softmax(dim=1)[0, class_id].detach().cpu().numpy()
    return prediction, probability


def export_prediction(predicted, model_affine, source_path, output_path):
    """Restore integer class IDs to the source grid with nearest-neighbor sampling."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    predicted = np.asarray(predicted)
    if predicted.ndim != 3 or not np.isfinite(predicted).all() or np.any(predicted < 0) \
            or not np.array_equal(predicted, np.round(predicted)):
        raise ValueError("Prediction must be a 3-D nonnegative integer label map")
    maximum = int(predicted.max())
    if maximum > np.iinfo(np.int32).max:
        raise ValueError("Prediction class IDs exceed int32 range")
    dtype = np.uint8 if maximum <= 255 else np.int16 if maximum <= 32767 else np.int32
    source = nib.load(str(source_path))
    if len(source.shape) != 3:
        raise ValueError("Prediction export requires a 3-D source image")
    model_image = nib.Nifti1Image(predicted.astype(dtype), model_affine)
    restored = resample_from_to(model_image, (source.shape, source.affine),
                                order=0, mode="constant", cval=0)
    data = np.asarray(restored.dataobj, dtype=dtype)
    header = source.header.copy()
    header.set_data_dtype(dtype)
    header.set_intent("label")
    header["cal_min"], header["cal_max"] = 0, maximum
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, source.affine, header), str(output_path))
    return output_path



class ConfiguredPredictor:
    """Checkpoint loading shared with the UI; inference parameters read from cfg."""

    def __init__(self, cfg):
        import torch
        import src.model  # register GCD models
        from src.utils import build_model
        from monai.inferers import SlidingWindowInferer
        self.runtime = ModelRuntimeLoader(build_model=build_model).load(cfg)
        self.cfg = cfg
        settings = cfg.inference
        self.inferer = SlidingWindowInferer(
            roi_size=tuple(settings.roi_size), sw_batch_size=int(settings.sw_batch_size),
            overlap=float(settings.overlap), mode=settings.blend_mode,
            sigma_scale=settings.get("sigma_scale", .125),
            padding_mode=settings.get("padding_mode", "constant"),
            cval=float(settings.get("cval", 0)),
            sw_device=self.runtime.device, device=self.runtime.device,
        )

    def logits(self, batch):
        import torch
        with torch.inference_mode():
            output = self.inferer(batch.to(self.runtime.device), self.runtime.model)
        return output[0] if isinstance(output, (tuple, list)) else output

    def readout(self, logits, class_id):
        if not 0 <= class_id < logits.shape[1]:
            raise ValueError(f"class {class_id} is not in model output")
        return prediction(logits, class_id, bool(self.cfg.get(
            "postprocessing", {}).get("keep_largest_connected_component", False)))

    def __call__(self, volume, class_id):
        import torch
        tensor = torch.as_tensor(volume, dtype=torch.float32, device=self.runtime.device)[None, None]
        return self.readout(self.logits(tensor), class_id)
