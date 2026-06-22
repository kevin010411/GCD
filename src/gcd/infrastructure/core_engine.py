from __future__ import annotations

import os
from copy import deepcopy
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..domain import DatasetInput
from .cam_methods import (
    CamMethod,
    CamPatchContext,
    GradCAMTestMethod,
    GradCamMethod,
    PerturbationOcclusionMethod,
    SaliencyMapMethod,
    XResCamMethod,
    XaiLayerSelection,
    XaiMethodRegistry,
)
from .layer_hooks import XaiLayerHookManager

if TYPE_CHECKING:
    import numpy as np
    import torch


def _config_from_file(config_path: str):
    from mmengine import Config

    return Config.fromfile(config_path)


def _build_model(cfg):
    import src.model  # noqa: F401
    from src.utils import build_model

    return build_model(cfg)


def _timer(*args, **kwargs):
    from src.utils.utils import timer

    return timer(*args, **kwargs)


class ModelLoadStateDictError(RuntimeError):
    def __init__(self, message: str, *, error_file: str) -> None:
        super().__init__(message)
        self.error_file = error_file
        self.skip_error_store = True


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None


def _layer_channel_counts(patch_payload: dict[str, object]) -> dict[str, int]:
    import torch

    layers = patch_payload.get("layers")
    if not isinstance(layers, dict):
        return {}
    counts: dict[str, int] = {}
    for name, layer_payload in layers.items():
        if not isinstance(layer_payload, dict):
            continue
        activation = layer_payload.get("activation")
        if isinstance(activation, torch.Tensor):
            counts[str(name)] = int(activation.size(1))
    return counts


class GradCamEngine:
    def __init__(
        self,
        cfg_path: str,
        save_dir: str | None = None,
        logger: Callable[[str], None] | None = None,
        error_store=None,
    ) -> None:
        self._logger = logger or (lambda message: None)
        self.error_store = error_store
        self.cfg = _config_from_file(cfg_path)
        self._apply_config()

        self.cam = None
        self.volume_data = None
        self.img0 = None
        self.img1 = None
        self.model_input_metadata = self._default_display_metadata()
        self.origin_img = None
        self.origin_meta = {}
        self.origin_shape = None
        self.img1_spacing = self.SPACING
        self.display_metadata = self._default_display_metadata()
        self.layers = {"layer1": 1}
        self.file_name = ""
        self.patch: list[dict[str, object]] = []
        self.target_class = 1
        self.active_objective_id = "predicted_target_mask"
        self.xai_cache_key = ""
        self.gradient_objectives = self._default_gradient_objectives()
        self.perturbation_objectives = self._default_perturbation_objectives()
        self.objectives = self.gradient_objectives
        self.xai_method_registry = XaiMethodRegistry.default(
            self._predicted_target_mask_objective
        )
        self.cam_methods: dict[str, CamMethod] = self.xai_method_registry.methods_by_id
        self.active_method_id = GradCamMethod.id

        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _apply_config(self) -> None:
        self.SIZE = self.cfg["size"]
        self.STRIDE = self.cfg["stride"]
        self.SPACING = self.cfg["spacing"]
        self.PERMUTE = self.cfg["permute"]

    def _log(self, message: str) -> None:
        self._logger(message)

    @staticmethod
    def _default_display_metadata() -> dict[str, object]:
        import numpy as np

        affine = np.eye(4, dtype=np.float32)
        direction = np.eye(3, dtype=np.float32)
        return {
            "affine": affine,
            "origin": (0.0, 0.0, 0.0),
            "spacing": (1.0, 1.0, 1.0),
            "direction": tuple(tuple(float(v) for v in row) for row in direction),
            "vtk_origin": (0.0, 0.0, 0.0),
            "vtk_spacing": (1.0, 1.0, 1.0),
            "vtk_direction": tuple(tuple(float(v) for v in row) for row in direction),
        }

    def set_config(self, config_path: str) -> None:
        self.cfg = _config_from_file(config_path)
        self._apply_config()
        self._log(f"已設定 Config 為: {config_path}")

    def model_layer_metadata(self) -> dict[str, object]:
        model = _build_model(self.cfg.model)
        hook_manager = XaiLayerHookManager(model)
        layer_names = list(hook_manager.layer_names)
        default_layer = str(self.cfg.get("default_layer", "") or "")
        selected_layer = (
            default_layer
            if default_layer in layer_names
            else (layer_names[0] if layer_names else "")
        )
        return {
            "layer_names": layer_names,
            "selected_layer": selected_layer,
            "feature_size": 0,
        }

    def set_target_class(self, target_class: int) -> None:
        self.target_class = int(target_class)

    def available_cam_methods(self, category: str | None = None) -> list[dict[str, object]]:
        family = "gradient" if category == "grad" else category
        if not hasattr(self, "xai_method_registry"):
            methods = self.cam_methods.values()
            if family is not None:
                methods = [
                    method
                    for method in methods
                    if getattr(method, "family", getattr(method, "category", "")) == family
                    or getattr(method, "category", "") == category
                ]
            return [
                {
                    "id": method.id,
                    "name": method.display_name,
                    "uses_layer_controls": bool(method.uses_layer_controls),
                    "uses_objective": bool(getattr(method, "uses_objective", True)),
                    "parameters": [
                        parameter.to_dict()
                        for parameter in getattr(method, "parameter_schema", lambda: ())()
                    ],
                }
                for method in methods
            ]
        return self.xai_method_registry.available_methods(family)

    def available_xai_method_families(self) -> list[dict[str, object]]:
        return self.xai_method_registry.available_families()

    def available_xai_methods(
        self, family: str | None = None
    ) -> list[dict[str, object]]:
        return self.xai_method_registry.available_methods(family)

    def default_feature_size(self) -> int:
        return list(self.layers.values())[0]

    def dataset_input(self) -> DatasetInput:
        return DatasetInput(
            img0=deepcopy(self.img0),
            img1=(
                deepcopy(self.img1)
                if self.origin_img is None and getattr(self, "img1", None) is not None
                else None
            ),
            origin_img=deepcopy(self.origin_img),
            origin_meta=deepcopy(self.origin_meta),
            origin_shape=deepcopy(self.origin_shape),
            img1_spacing=deepcopy(self.img1_spacing),
            display_metadata=deepcopy(self.display_metadata),
            layers=deepcopy(self.layers),
            file_name=self.file_name,
            target_class=self.target_class,
            active_method_id=self.active_method_id,
            active_objective_id=getattr(
                self, "active_objective_id", "predicted_target_mask"
            ),
            xai_cache_key=self.xai_cache_key,
            raw_display_data=deepcopy(self.volume_data),
            raw_spacing=deepcopy(self.display_metadata.get("spacing", self.img1_spacing)),
            raw_display_metadata=deepcopy(self.display_metadata),
        )

    def load_dataset_input(self, dataset_input: DatasetInput) -> None:
        self.cam = None
        self.volume_data = None
        self.img0 = deepcopy(dataset_input.img0)
        self.img1 = None
        self.origin_img = deepcopy(dataset_input.origin_img)
        self.origin_meta = deepcopy(dataset_input.origin_meta)
        self.origin_shape = deepcopy(dataset_input.origin_shape)
        self.img1_spacing = deepcopy(dataset_input.img1_spacing)
        self.display_metadata = deepcopy(
            dataset_input.raw_display_metadata or dataset_input.display_metadata
        )
        self.model_input_metadata = deepcopy(dataset_input.display_metadata)
        self.layers = deepcopy(dataset_input.layers)
        self.file_name = str(dataset_input.file_name)
        self.patch = []
        self.target_class = int(dataset_input.target_class)
        self.active_method_id = str(dataset_input.active_method_id)
        self.active_objective_id = str(
            getattr(dataset_input, "active_objective_id", "predicted_target_mask")
        )
        self.model_output = None
        self.xai_cache_key = str(dataset_input.xai_cache_key)
        self.volume_data = deepcopy(dataset_input.raw_display_data)
        if self.volume_data is None and dataset_input.img1 is not None:
            self.img1 = deepcopy(dataset_input.img1)
            self.volume_data = self.img1[0].permute(*self.PERMUTE)

    def _resolve_cam_method(self, method: str | None) -> CamMethod:
        requested = (method or self.active_method_id or GradCamMethod.id).strip().lower()
        if requested in self.cam_methods:
            if hasattr(self, "xai_method_registry"):
                return self.xai_method_registry.resolve(requested)
            return self.cam_methods[requested]
        self._log(f"未知 CAM method '{method}'，改用預設方法: {GradCamMethod.id}")
        if hasattr(self, "xai_method_registry"):
            return self.xai_method_registry.resolve(GradCamMethod.id)
        return self.cam_methods[GradCamMethod.id]


    @staticmethod
    def _predicted_target_mask_objective(
        logits: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        import torch

        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        index = torch.argmax(logits[0], dim=0)
        loss = (logits[0, target_class] * (index == target_class)).sum()
        return loss

    @staticmethod
    def _target_logit_sum_objective(logits: torch.Tensor, target_class: int) -> torch.Tensor:
        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        return logits[0, target_class].sum()

    _gradcam_objective = _predicted_target_mask_objective

    @staticmethod
    def _target_probability_sum_objective(
        logits: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        import torch

        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        return torch.softmax(logits[0], dim=0)[target_class].sum()

    @staticmethod
    def _target_margin_objective(logits: torch.Tensor, target_class: int) -> torch.Tensor:
        import torch

        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        target_logits = logits[0, target_class]
        other_logits = torch.cat(
            [logits[0, :target_class], logits[0, target_class + 1 :]], dim=0
        )
        return (target_logits - torch.amax(other_logits, dim=0)).sum()

    def available_objectives(self, family: str | None = None) -> list[dict[str, object]]:
        objectives = self._objectives_for_family(family)
        return [
            {"id": objective_id, "name": label}
            for objective_id, (label, _objective) in objectives.items()
        ]

    def _default_gradient_objectives(
        self,
    ) -> dict[str, tuple[str, Callable[[torch.Tensor, int], torch.Tensor]]]:
        return {
            "predicted_target_mask": (
                "Predicted Target Mask",
                self._predicted_target_mask_objective,
            ),
            "target_logit_sum": ("Target Logit Sum", self._target_logit_sum_objective),
            "target_probability_sum": (
                "Target Probability Sum",
                self._target_probability_sum_objective,
            ),
            "target_margin": ("Target Margin", self._target_margin_objective),
        }

    def _default_perturbation_objectives(
        self,
    ) -> dict[str, tuple[str, Callable[..., torch.Tensor]]]:
        return {
            "predicted_mask_dice": (
                "Predicted Mask Dice",
                self._predicted_mask_dice_score,
            ),
            "predicted_mask_iou": (
                "Predicted Mask IoU",
                self._predicted_mask_iou_score,
            ),
            "target_probability_sum": (
                "Target Probability Sum",
                self._target_probability_sum_objective,
            ),
            "target_logit_sum": ("Target Logit Sum", self._target_logit_sum_objective),
        }

    @staticmethod
    def _predicted_mask_dice_score(
        logits: torch.Tensor, target_class: int, reference_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        import torch

        if reference_mask is None:
            reference_mask = torch.argmax(logits[0], dim=0) == target_class
        pred_mask = torch.argmax(logits[0], dim=0) == target_class
        reference_mask = reference_mask.to(device=pred_mask.device, dtype=torch.bool)
        intersection = torch.logical_and(pred_mask, reference_mask).sum(dtype=torch.float32)
        denom = pred_mask.sum(dtype=torch.float32) + reference_mask.sum(dtype=torch.float32)
        if denom == 0:
            return torch.ones((), device=logits.device, dtype=torch.float32)
        return (2.0 * intersection) / denom

    @staticmethod
    def _predicted_mask_iou_score(
        logits: torch.Tensor, target_class: int, reference_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        import torch

        if reference_mask is None:
            reference_mask = torch.argmax(logits[0], dim=0) == target_class
        pred_mask = torch.argmax(logits[0], dim=0) == target_class
        reference_mask = reference_mask.to(device=pred_mask.device, dtype=torch.bool)
        intersection = torch.logical_and(pred_mask, reference_mask).sum(dtype=torch.float32)
        union = torch.logical_or(pred_mask, reference_mask).sum(dtype=torch.float32)
        if union == 0:
            return torch.ones((), device=logits.device, dtype=torch.float32)
        return intersection / union

    def _objectives_for_family(
        self, family: str | None
    ) -> dict[str, tuple[str, Callable[..., torch.Tensor]]]:
        if not hasattr(self, "gradient_objectives"):
            self.gradient_objectives = self._default_gradient_objectives()
        if not hasattr(self, "perturbation_objectives"):
            self.perturbation_objectives = self._default_perturbation_objectives()
        if family == "perturbation":
            return self.perturbation_objectives
        return self.gradient_objectives

    def _resolve_objective(
        self, objective_id: str | None, family: str | None = None
    ) -> tuple[str, Callable[..., torch.Tensor]]:
        objectives = self._objectives_for_family(family)
        active_objective_id = getattr(
            self, "active_objective_id", "predicted_target_mask"
        )
        requested = (objective_id or active_objective_id).strip().lower()
        if requested in objectives:
            return requested, objectives[requested][1]
        fallback = "predicted_mask_dice" if family == "perturbation" else "predicted_target_mask"
        self._log(f"未知 objective '{objective_id}'，改用預設 objective: {fallback}")
        return fallback, objectives[fallback][1]

    def load_volume(self, input_file: str | None = None) -> list[str]:
        import monai.transforms as mt
        import torch

        messages: list[str] = []
        if input_file is not None:
            self.file_name = input_file
        if not self.file_name:
            raise ValueError("尚未指定輸入檔案，無法載入與處理資料。")

        self.origin_img, self.origin_meta = mt.LoadImage(image_only=False)(
            self.file_name
        )
        self.img0 = mt.EnsureChannelFirst()(self.origin_img, self.origin_meta)

        with _timer("載入展示資料"):
            try:
                self.origin_meta = dict(self.img0.meta)
            except Exception:
                self.origin_meta = {}
            self.origin_shape = tuple(self.img0.shape)
            raw_affine = self._extract_affine(self.img0)
            self.display_metadata = self._build_display_metadata(raw_affine)
            self.display_metadata.update(
                {
                    "source_affine": self._safe_affine(),
                    "source_shape": tuple(int(v) for v in self.origin_shape[1:]),
                    "display_permute": tuple(int(v) for v in self.PERMUTE),
                }
            )
            self.volume_data = self.img0[0].permute(*self.PERMUTE).to(torch.float32)
            self.cam = torch.zeros_like(self.volume_data)
            self.img1 = None
            self.img1_spacing = (
                self.SPACING[self.PERMUTE[0]],
                self.SPACING[self.PERMUTE[1]],
                self.SPACING[self.PERMUTE[2]],
            )
            self.model_input_metadata = self._default_display_metadata()
            self.model_output = None
            self.patch = []
            self.layers = {self.cfg["default_layer"]: 1}
            self.xai_cache_key = ""

        for message in messages:
            self._log(message)
        return messages

    def prepare_model_input(self) -> list[str]:
        if getattr(self, "img0", None) is None:
            if getattr(self, "img1", None) is not None:
                self.model_input_metadata = deepcopy(
                    getattr(self, "display_metadata", self._default_display_metadata())
                )
                return []
            if getattr(self, "origin_img", None) is None:
                raise ValueError("缺少原始資料，無法產生模型輸入。")
            import monai.transforms as mt

            self.img0 = mt.EnsureChannelFirst()(self.origin_img, self.origin_meta)
            try:
                self.origin_meta = dict(self.img0.meta)
            except Exception:
                self.origin_meta = dict(self.origin_meta or {})
            self.origin_shape = tuple(self.img0.shape)

        import monai.transforms as mt
        import torch

        messages: list[str] = []
        with _timer("資料前處理"):
            img1 = mt.Spacing(mode="bilinear", pixdim=self.SPACING)(self.img0)
            width, depth = self.SIZE + self.STRIDE, self.SIZE
            img1 = mt.SpatialPad(
                spatial_size=(width, width, depth), mode="constant", value=0
            )(img1)
            img1_affine = self._extract_affine(img1)

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
                img1_affine = self._shift_affine_for_padding(img1_affine, pad_offsets)

            img1 = mt.ScaleIntensityRange(
                a_min=-42, a_max=423, b_min=0, b_max=1, clip=True
            )(img1)
            self.img1 = img1
            self.img1_spacing = (
                self.SPACING[self.PERMUTE[0]],
                self.SPACING[self.PERMUTE[1]],
                self.SPACING[self.PERMUTE[2]],
            )
            self.model_input_metadata = self._build_display_metadata(img1_affine)
            self.model_input_metadata.update(
                {
                    "source_affine": self._safe_affine(),
                    "source_shape": tuple(int(v) for v in self.origin_shape[1:]),
                    "display_permute": tuple(int(v) for v in self.PERMUTE),
                }
            )
        return messages

    def prepare_xai_inputs(
        self,
        method: str | None = None,
        objective_id: str | None = None,
        method_params: dict[str, object] | None = None,
    ) -> None:
        import torch

        if not self.file_name:
            raise ValueError("尚未載入檔案，無法準備 XAI 輸入。")
        self.prepare_model_input()
        cam_method = self._resolve_cam_method(method)
        selected_objective_id, objective = self._resolve_objective(
            objective_id, cam_method.family
        )
        self.active_method_id = cam_method.id
        self.active_objective_id = selected_objective_id
        self.patch = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with _timer("載入模型"):
            model = _build_model(self.cfg.model).to(device)
            ckpt_path = self.cfg.ckpt
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            pth = torch.load(ckpt_path, map_location="cpu")
            sd = pth["state_dict"].copy() if "state_dict" in pth else pth.copy()

            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                error_path = None
                if self.error_store is not None:
                    error_path = self.error_store.save_json(
                        {
                            "error_type": "model_load_error",
                            "config_path": getattr(self.cfg, "filename", None),
                            "checkpoint_path": ckpt_path,
                            "missing_count": len(missing),
                            "missing_keys": list(missing),
                            "unexpected_count": len(unexpected),
                            "unexpected_keys": list(unexpected),
                        },
                        suffix="model_load_error",
                    )
                raise ModelLoadStateDictError(
                    (
                        "Model load failed because checkpoint keys do not match the model."
                        + (
                            f" Error details saved to: {error_path}"
                            if error_path
                            else ""
                        )
                    ),
                    error_file=str(error_path) if error_path else "",
                )
            model.eval()

        img2 = self.img1.unsqueeze(0).to(device)
        img2.requires_grad_()

        hook_manager = (
            XaiLayerHookManager(model) if cam_method.uses_layer_controls else None
        )

        perturb_reference_volume = self._perturb_reference_volume(
            method_params or {}, device
        )

        with _timer("模型推論", track_gpu=True):
            x0, y0, z0 = list(
                (
                    torch.tensor(self.img1[0].shape)
                    - torch.tensor(
                        [self.STRIDE + self.SIZE, self.STRIDE + self.SIZE, self.SIZE]
                    )
                )
                // 2
            )

            self.patch = []
            tiles = [
                (0, 0),
                (0, self.STRIDE),
                (self.STRIDE, 0),
                (self.STRIDE, self.STRIDE),
            ]
            tile_progress_units = self._perturb_progress_units(
                cam_method.id, method_params or {}
            )
            progress_total = tile_progress_units * len(tiles)
            progress_callback = (method_params or {}).get("_progress_callback")
            if cam_method.family == "perturbation" and callable(progress_callback):
                progress_callback({"current": 0, "total": progress_total})
            hook_context = hook_manager if hook_manager is not None else _NullContext()
            with hook_context:
                for x, y in tiles:
                    self._wait_for_perturbation_pause(cam_method, method_params or {})
                    model.zero_grad(set_to_none=True)
                    if img2.grad is not None:
                        img2.grad = None
                    if hook_manager is not None:
                        hook_manager.clear()
                    tile_input = img2[
                        ...,
                        x0 + x : x0 + x + self.SIZE,
                        y0 + y : y0 + y + self.SIZE,
                        z0 : z0 + self.SIZE,
                    ]
                    tile_input.retain_grad()
                    logits = model(tile_input)
                    layers_by_name = (
                        hook_manager.layers_by_name()
                        if hook_manager is not None
                        else {}
                    )
                    tile_method_params = dict(method_params or {})
                    tile_method_params["_score_logger"] = self._log
                    tile_method_params["_tile_index"] = len(self.patch)
                    tile_method_params["_progress_total"] = progress_total
                    tile_method_params["_progress_offset"] = (
                        len(self.patch) * tile_progress_units
                    )
                    tile_method_params["_preview_spacing"] = getattr(
                        self, "img1_spacing", getattr(self, "SPACING", (1.0, 1.0, 1.0))
                    )
                    tile_method_params["_preview_metadata"] = {
                        **self.model_input_metadata,
                        "volume_id": "perturb-preview",
                    }
                    tile_method_params["_preview_full_input"] = img2[0].detach()
                    tile_method_params["_preview_tile_origin"] = (
                        int(x0 + x),
                        int(y0 + y),
                        int(z0),
                    )
                    tile_method_params["_preview_permute"] = tuple(
                        getattr(self, "PERMUTE", (0, 1, 2))
                    )
                    if perturb_reference_volume is not None:
                        tile_method_params["reference_mask"] = perturb_reference_volume[
                            x0 + x : x0 + x + self.SIZE,
                            y0 + y : y0 + y + self.SIZE,
                            z0 : z0 + self.SIZE,
                        ]

                    self._wait_for_perturbation_pause(cam_method, tile_method_params)
                    self.patch.append(
                        cam_method.collect_patch_data(
                            CamPatchContext(
                                input_tensor=tile_input,
                                logits=logits,
                                layers_by_name=layers_by_name,
                                target_class=self.target_class,
                                objective=objective,
                                model=model,
                                method_params=tile_method_params,
                                device=device,
                            )
                        )
                    )

            self.layers = (
                _layer_channel_counts(self.patch[-1])
                if hook_manager is not None
                else {"input": 1}
            )
            self.model_output = logits.detach().to("cpu")

            del model, img2, pth, sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cfg_name = str(getattr(self.cfg, "filename", "") or "")
        self.xai_cache_key = (
            f"{cfg_name}|{self.target_class}|{cam_method.id}|{selected_objective_id}"
        )

    def _perturb_reference_volume(
        self, method_params: dict[str, object], device
    ) -> torch.Tensor | None:
        import torch

        answer_data = method_params.get("answer_data")
        if answer_data is None or self.img1 is None:
            return None
        data = torch.as_tensor(answer_data, device=device).to(torch.float32)
        data = data.squeeze()
        if data.ndim != 3:
            return None
        target_shape = tuple(int(v) for v in self.img1[0].shape)
        display_shape = self._display_space_shape()
        if display_shape is not None and tuple(data.shape) == display_shape:
            data = self._inv_permute(data)
        if tuple(data.shape) != target_shape:
            data = self._resize(data, target_shape)

        target_class = int(self.target_class)
        finite = data[torch.isfinite(data)]
        if finite.numel() == 0:
            return torch.zeros(target_shape, dtype=torch.bool, device=device)
        rounded = torch.round(data)
        if torch.max(finite) > 1.5:
            return rounded == target_class
        return data > 0.5

    def _perturb_progress_units(
        self, method: str, method_params: dict[str, object]
    ) -> int:
        if method == "perturb_occlusion":
            block_size = int(method_params.get("block_size", 16) or 16)
            stride = int(method_params.get("stride", 8) or 8)
            starts = self._perturb_axis_count(self.SIZE, block_size, stride)
            return starts**3
        if method == "perturb_lime":
            return max(1, int(method_params.get("num_samples", 128) or 128))
        if method == "perturb_rise":
            return max(1, int(method_params.get("num_masks", 64) or 64))
        return 1

    @staticmethod
    def _wait_for_perturbation_pause(cam_method, method_params: dict[str, object]) -> None:
        if getattr(cam_method, "family", "") != "perturbation":
            return
        controller = method_params.get("_pause_controller")
        waiter = getattr(controller, "wait_if_paused", None)
        if callable(waiter):
            waiter()

    @staticmethod
    def _perturb_axis_count(size: int, block_size: int, stride: int) -> int:
        size = max(1, int(size))
        block_size = max(1, min(int(block_size), size))
        stride = max(1, int(stride))
        if block_size >= size:
            return 1
        count = ((size - block_size) // stride) + 1
        last = size - block_size
        if (count - 1) * stride != last:
            count += 1
        return count

    def _display_space_shape(self) -> tuple[int, int, int] | None:
        volume_data = getattr(self, "volume_data", None)
        if volume_data is not None:
            return tuple(int(v) for v in volume_data.shape)
        if getattr(self, "img1", None) is None:
            return None
        return tuple(int(v) for v in self.img1[0].permute(*self.PERMUTE).shape)

    def load_and_process_input(
        self, input_file: str | None = None, method: str | None = None
    ) -> list[str]:
        messages = self.load_volume(input_file)
        self.prepare_xai_inputs(method)
        return messages

    def compute_cam(
        self,
        layer: str | None = None,
        n1: int = 0,
        n2: int = 999,
        method: str | None = None,
        method_params: dict[str, object] | None = None,
    ) -> str:
        import torch
        import torch.nn.functional as F

        if not self.file_name:
            raise ValueError("尚未載入檔案，無法計算 CAM。")
        cam_method = self._resolve_cam_method(method)
        if self.patch and cam_method.id != self.active_method_id:
            raise ValueError(
                "目前的 CAM patch 資料與指定 method 不一致，請重新載入輸入資料後再計算。"
            )
        if not self.patch:
            raise RuntimeError("尚未準備 XAI patch 資料，請先執行 prepare_xai_inputs。")
        if any(
            not isinstance(item, dict) or item.get("method") != cam_method.id
            for item in self.patch
        ):
            raise ValueError("目前的 CAM patch payload 與指定 method 不一致。")
        self.active_method_id = cam_method.id

        if cam_method.uses_layer_controls:
            available_layers = list(self.layers.keys())
            selected_layer = layer or self.cfg["default_layer"]
            if selected_layer not in self.layers:
                fallback_layer = (
                    self.cfg["default_layer"]
                    if self.cfg["default_layer"] in self.layers
                    else (available_layers[0] if available_layers else "")
                )
                if not fallback_layer:
                    raise RuntimeError("目前沒有可用的 CAM layer。")
                selected_layer = fallback_layer
                self._log(f"指定 layer 不存在，改用可用 layer: {selected_layer}")
        else:
            selected_layer = "input"
            self.layers = {"input": 1}

        with _timer(f"layer={selected_layer} 計算 GradCAM"):
            n2 = min(n2, self.layers[selected_layer])
            shape = list(self.img1[0].shape)
            cam = torch.zeros(shape, dtype=torch.float32)
            if not isinstance(self.patch[0].get("pred"), torch.Tensor):
                raise TypeError("CAM patch payload 缺少 pred tensor。")
            pred_shape = list(self.patch[0]["pred"].shape)[:2] + shape
            model_out = torch.zeros(pred_shape, dtype=torch.float32)

            x0, y0, z0 = list(
                (
                    torch.tensor(self.img1[0].shape)
                    - torch.tensor(
                        [self.STRIDE + self.SIZE, self.STRIDE + self.SIZE, self.SIZE]
                    )
                )
                // 2
            )

            tiles = [
                (0, 0),
                (0, self.STRIDE),
                (self.STRIDE, 0),
                (self.STRIDE, self.STRIDE),
            ]
            for index, (x, y) in enumerate(tiles):
                q = cam_method.build_tile_cam(
                    self.patch[index],
                    XaiLayerSelection(
                        selected_layer,
                        n1,
                        n2,
                        (self.SIZE, self.SIZE, self.SIZE),
                    ),
                    method_params=method_params,
                )

                p1 = self.patch[index]["pred"]
                if not isinstance(p1, torch.Tensor):
                    raise TypeError("CAM patch payload 缺少 pred tensor。")
                p1 = F.interpolate(
                    p1, size=(self.SIZE, self.SIZE, self.SIZE), mode="trilinear"
                )

                overlap = self.SIZE - self.STRIDE
                if index in (0, 1):
                    for i in range(overlap):
                        weight = (overlap - i) / overlap
                        q[0, 0, self.STRIDE + i, :, :] *= weight
                        p1[0, 0, self.STRIDE + i, :, :] *= weight
                if index in (2, 3):
                    for i in range(overlap):
                        weight = i / overlap
                        q[0, 0, i, :, :] *= weight
                        p1[0, 0, i, :, :] *= weight
                if index in (0, 2):
                    for i in range(overlap):
                        weight = (overlap - i) / overlap
                        q[0, 0, :, self.STRIDE + i, :] *= weight
                        p1[0, 0, :, self.STRIDE + i, :] *= weight
                if index in (1, 3):
                    for i in range(overlap):
                        weight = i / overlap
                        q[0, 0, :, i, :] *= weight
                        p1[0, 0, :, i, :] *= weight

                xs = slice(x0 + x, x0 + x + self.SIZE)
                ys = slice(y0 + y, y0 + y + self.SIZE)
                zs = slice(z0, z0 + self.SIZE)

                cam[xs, ys, zs] += q[0, 0]
                model_out[:, :, xs, ys, zs] += p1

            perturb_signal_max = None
            if cam_method.family == "perturbation":
                cam = torch.abs(cam)
                perturb_signal_max = torch.max(cam)
            else:
                cam = torch.maximum(cam, torch.tensor(0))
            cam -= torch.min(cam)
            maximum = torch.max(cam)
            if maximum > 0:
                cam /= maximum
            elif (
                cam_method.family == "perturbation"
                and perturb_signal_max is not None
                and perturb_signal_max > 0
            ):
                cam = torch.ones_like(cam)

            self.cam = cam.permute(*self.PERMUTE)
            if getattr(self, "volume_data", None) is None:
                self.volume_data = self.img1[0].permute(*self.PERMUTE)
            self.model_output = torch.argmax(model_out, dim=1)[0].permute(*self.PERMUTE)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.file_name))[0]
            self._save_volume(
                cam.permute(*self.PERMUTE),
                os.path.join(
                    self.save_dir, f"{base}_{self.cfg.model.type}_{selected_layer}_cam"
                ),
            )
            self._save_volume(
                self.model_output,
                os.path.join(self.save_dir, f"{base}_{self.cfg.model.type}_pred"),
            )
            self._save_volume(
                self.volume_data, os.path.join(self.save_dir, f"{base}_img")
            )
        return selected_layer

    def _inv_permute(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return tensor
        inverse = [0, 0, 0]
        for index, permute_index in enumerate(self.PERMUTE):
            inverse[permute_index] = index
        return tensor.permute(*inverse)

    def _resize(self, volume: torch.Tensor, size) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        mode = (
            "trilinear"
            if volume.dtype in (torch.float32, torch.float16, torch.float64)
            else "nearest"
        )
        volume = volume.to(torch.float32)
        data = volume.unsqueeze(0).unsqueeze(0)
        if mode == "trilinear":
            data = F.interpolate(data, size=size, mode=mode, align_corners=False)
        else:
            data = F.interpolate(data, size=size, mode=mode)
        return data[0, 0]

    def _to_origin_space(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return tensor
        volume = self._inv_permute(tensor)
        want = (
            list(self.origin_shape[1:])
            if self.origin_shape is not None
            else list(volume.shape)
        )
        return (
            self._resize(volume, want)
            if tuple(volume.shape) != tuple(want)
            else volume.to(torch.float32)
        )

    def to_raw_display_space(self, tensor):
        import torch

        if tensor is None:
            return tensor
        data = torch.as_tensor(tensor).to(torch.float32)
        source_space = self._to_origin_space(data)
        return source_space.permute(*self.PERMUTE)

    def _safe_affine(self):
        import numpy as np

        spacing = None
        if (
            self.origin_meta
            and "pixdim" in self.origin_meta
            and len(self.origin_meta["pixdim"]) >= 4
        ):
            try:
                spacing = tuple(map(float, self.origin_meta["pixdim"][1:4]))
            except Exception:
                spacing = None
        if spacing is None:
            spacing = tuple(
                float(value) for value in (self.img1_spacing or (1.0, 1.0, 1.0))
            )

        affine = (
            self.origin_meta["affine"]
            if self.origin_meta and "affine" in self.origin_meta
            else None
        )
        if affine is None:
            affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype(
                "float32"
            )
        return affine

    def _extract_affine(self, image) -> np.ndarray:
        import numpy as np

        meta = getattr(image, "meta", None)
        if meta is not None:
            affine = meta.get("affine")
            if affine is not None:
                return np.array(affine, dtype=np.float32, copy=True)
        return np.array(self._safe_affine(), dtype=np.float32, copy=True)

    @staticmethod
    def _shift_affine_for_padding(
        affine: np.ndarray, offsets: list[int] | tuple[int, int, int]
    ) -> np.ndarray:
        import numpy as np

        shifted = np.array(affine, dtype=np.float32, copy=True)
        offset_vector = np.array(offsets, dtype=np.float32)
        shifted[:3, 3] -= shifted[:3, :3] @ offset_vector
        return shifted

    def _build_display_metadata(self, affine: np.ndarray) -> dict[str, object]:
        import numpy as np

        axis_order = list(self.PERMUTE)
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
            "vtk_direction": tuple(
                tuple(float(v) for v in row) for row in vtk_direction.T
            ),
        }

    def _save_volume(
        self, tensor: torch.Tensor, stem: str, exist_ok: bool = True
    ) -> None:
        if tensor is None:
            return
        try:
            import nibabel as nib

            array = (
                self._to_origin_space(tensor).detach().cpu().numpy().astype("float32")
            )
            affine = self._safe_affine()
            nii = nib.Nifti1Image(array, affine=affine)
            if exist_ok:
                nib.save(nii, f"{stem}.nii.gz")
            elif os.path.exists(f"{stem}.nii.gz"):
                raise FileExistsError(f"File({stem}.nii.gz) already exists.")
        except Exception as exc:
            self._log(f"note: nibabel not available or failed ({exc})")
