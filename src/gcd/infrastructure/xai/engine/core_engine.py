from __future__ import annotations

import os
from copy import deepcopy
from collections.abc import Callable
from typing import TYPE_CHECKING

from ....domain import DatasetInput
from ..methods.cam_methods import (
    CamMethod,
    GradCamMethod,
    XaiMethodRegistry,
)
from ..runtime.layer_hooks import XaiLayerHookManager
from ...model_input_preprocessor import (
    ModelInputPreprocessConfig,
    ModelInputPreprocessor,
)
from ..runtime.model_runtime_loader import ModelLoadStateDictError, ModelRuntimeLoader
from ..tiling.tile_collector import TileCollectionRequest, TileCollector
from ..tiling.tile_strategy import TileStrategyResolver
from ...volume_loading import (
    VolumeLoadConfig,
    VolumeLoadingService,
    safe_affine,
)
from ..runners.xai_cam_runner import XaiCamRunRequest, XaiCamRunner

if TYPE_CHECKING:
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
        self.model_input = None
        self.model_input_metadata = self._default_display_metadata()
        self.origin_img = None
        self.origin_meta = {}
        self.origin_shape = None
        self.img1_spacing = self.SPACING
        self.display_metadata = self._default_display_metadata()
        self.layers = {"layer1": 1}
        self.file_name = ""
        self.patch: list[dict[str, object]] = []
        self.tile_plan = None
        self.target_class = 1
        self.active_objective_id = "predicted_target_mask"
        self.xai_cache_key = ""
        self.gradient_objectives = self._default_gradient_objectives()
        self.perturbation_objectives = self._default_perturbation_objectives()
        self.xai_method_registry = XaiMethodRegistry.default(
            self._predicted_target_mask_objective
        )
        self.cam_methods: dict[str, CamMethod] = self.xai_method_registry.methods_by_id
        self.active_method_id = GradCamMethod.id
        self.volume_loader = VolumeLoadingService()
        self.model_input_preprocessor = ModelInputPreprocessor()
        self.model_runtime_loader = ModelRuntimeLoader(
            build_model=_build_model,
            error_store=self.error_store,
        )
        self.tile_strategy_resolver = TileStrategyResolver()
        self.tile_collector = TileCollector()
        self.xai_cam_runner = XaiCamRunner()

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

    def _volume_load_config(self) -> VolumeLoadConfig:
        return VolumeLoadConfig(
            spacing=tuple(float(v) for v in self.SPACING),
            permute=tuple(int(v) for v in self.PERMUTE),
            default_layer=str(self.cfg["default_layer"]),
        )

    def _model_input_preprocess_config(self) -> ModelInputPreprocessConfig:
        return ModelInputPreprocessConfig(
            size=int(self.SIZE),
            stride=int(self.STRIDE),
            spacing=tuple(float(v) for v in self.SPACING),
            permute=tuple(int(v) for v in self.PERMUTE),
        )

    def _volume_loading_service(self) -> VolumeLoadingService:
        service = getattr(self, "volume_loader", None)
        if service is None:
            service = VolumeLoadingService()
            self.volume_loader = service
        return service

    def _model_input_service(self) -> ModelInputPreprocessor:
        service = getattr(self, "model_input_preprocessor", None)
        if service is None:
            service = ModelInputPreprocessor()
            self.model_input_preprocessor = service
        return service

    def _xai_runner(self) -> XaiCamRunner:
        runner = getattr(self, "xai_cam_runner", None)
        if runner is None:
            runner = XaiCamRunner()
            self.xai_cam_runner = runner
        return runner

    def _model_runtime_service(self) -> ModelRuntimeLoader:
        service = getattr(self, "model_runtime_loader", None)
        if service is None:
            service = ModelRuntimeLoader(
                build_model=_build_model,
                error_store=getattr(self, "error_store", None),
            )
            self.model_runtime_loader = service
        service._build_model = _build_model
        service.error_store = getattr(self, "error_store", None)
        return service

    def _tile_strategy_service(self) -> TileStrategyResolver:
        service = getattr(self, "tile_strategy_resolver", None)
        if service is None:
            service = TileStrategyResolver()
            self.tile_strategy_resolver = service
        return service

    def _tile_collector_service(self) -> TileCollector:
        service = getattr(self, "tile_collector", None)
        if service is None:
            service = TileCollector()
            self.tile_collector = service
        return service

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
        self.model_input = None
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
        self.tile_plan = None
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
        import torch

        if input_file is not None:
            self.file_name = input_file
        if not self.file_name:
            raise ValueError("尚未指定輸入檔案，無法載入與處理資料。")

        with _timer("載入展示資料"):
            loaded = self._volume_loading_service().load(
                self.file_name, self._volume_load_config()
            )
            self.origin_img = loaded.origin_img
            self.origin_meta = loaded.origin_meta
            self.origin_shape = loaded.origin_shape
            self.img0 = loaded.img0
            self.display_metadata = loaded.display_metadata
            self.volume_data = loaded.display_volume
            self.cam = torch.zeros_like(self.volume_data)
            self.img1 = None
            self.model_input = None
            self.img1_spacing = loaded.display_spacing
            self.model_input_metadata = self._default_display_metadata()
            self.model_output = None
            self.patch = []
            self.tile_plan = None
            self.layers = loaded.default_layers
            self.xai_cache_key = ""

        for message in loaded.messages:
            self._log(message)
        return list(loaded.messages)

    def prepare_model_input(self) -> list[str]:
        if getattr(self, "img0", None) is None and getattr(self, "img1", None) is not None:
            self.model_input_metadata = deepcopy(
                getattr(self, "display_metadata", self._default_display_metadata())
            )
            return []
        with _timer("資料前處理"):
            result = self._model_input_service().preprocess(
                img0=getattr(self, "img0", None),
                origin_img=getattr(self, "origin_img", None),
                origin_meta=getattr(self, "origin_meta", {}),
                origin_shape=getattr(self, "origin_shape", None),
                config=self._model_input_preprocess_config(),
            )
            self.img0 = result.img0
            self.img1 = result.img1
            self.model_input = result
            self.origin_meta = result.origin_meta
            self.origin_shape = result.origin_shape
            self.img1_spacing = result.img1_spacing
            self.model_input_metadata = result.model_input_metadata
        return list(result.messages)

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
        if self.img1 is None:
            raise ValueError("尚未產生模型輸入，無法準備 XAI 輸入。")
        cam_method = self._resolve_cam_method(method)
        selected_objective_id, objective = self._resolve_objective(
            objective_id, cam_method.family
        )
        self.active_method_id = cam_method.id
        self.active_objective_id = selected_objective_id
        self.patch = []
        self.tile_plan = None

        with _timer("載入模型"):
            runtime = self._model_runtime_service().load(self.cfg)

        track_gpu = getattr(runtime.device, "type", "") == "cuda"
        with _timer("模型推論", track_gpu=track_gpu, device=str(runtime.device)):
            tile_strategy = self._tile_strategy_service().resolve(method_params or {})
            tile_plan = tile_strategy.plan(
                input_shape=tuple(int(v) for v in self.img1[0].shape),
                patch_size=int(self.SIZE),
                stride=int(self.STRIDE),
            )
            perturb_reference_volume = self._perturb_reference_volume(
                method_params or {}, runtime.device
            )
            collection = self._tile_collector_service().collect(
                TileCollectionRequest(
                    model=runtime.model,
                    device=runtime.device,
                    model_input=self.img1,
                    method=cam_method,
                    objective=objective,
                    target_class=self.target_class,
                    tile_plan=tile_plan,
                    method_params=method_params or {},
                    model_input_spacing=getattr(
                        self, "img1_spacing", getattr(self, "SPACING", (1.0, 1.0, 1.0))
                    ),
                    model_input_metadata=getattr(self, "model_input_metadata", {}),
                    display_permute=tuple(int(v) for v in getattr(self, "PERMUTE", (0, 1, 2))),
                    reference_mask=perturb_reference_volume,
                    progress_units=self._perturb_progress_units,
                    pause_waiter=self._wait_for_perturbation_pause,
                    logger=self._log,
                )
            )
            self.patch = collection.patches
            self.layers = collection.layers
            self.model_output = collection.model_output
            self.tile_plan = collection.tile_plan
            del runtime
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

    def run_xai_method(
        self,
        layer: str | None = None,
        n1: int = 0,
        n2: int = 999,
        method: str | None = None,
        method_params: dict[str, object] | None = None,
    ) -> str:
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

        with _timer(f"method={cam_method.id} 計算 XAI CAM"):
            result = self._xai_runner().run(
                XaiCamRunRequest(
                    method=cam_method,
                    patches=self.patch,
                    layers=self.layers,
                    img1=self.img1,
                    size=int(self.SIZE),
                    stride=int(self.STRIDE),
                    permute=tuple(int(v) for v in self.PERMUTE),
                    default_layer=str(self.cfg["default_layer"]),
                    tile_plan=getattr(self, "tile_plan", None),
                    layer=layer,
                    n1=n1,
                    n2=n2,
                    method_params=method_params,
                    logger=self._log,
                )
            )
            selected_layer = result.selected_layer
            self.layers = result.layers
            self.cam = result.cam
            if getattr(self, "volume_data", None) is None:
                self.volume_data = self.img1[0].permute(*self.PERMUTE)
            self.model_output = result.model_output

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.file_name))[0]
            self._save_volume(
                self.cam,
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

    def compute_cam(
        self,
        layer: str | None = None,
        n1: int = 0,
        n2: int = 999,
        method: str | None = None,
        method_params: dict[str, object] | None = None,
    ) -> str:
        return self.run_xai_method(
            layer=layer,
            n1=n1,
            n2=n2,
            method=method,
            method_params=method_params,
        )

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

    def prediction_to_raw_display_space(self, tensor):
        """Restore a categorical prediction volume without interpolating class IDs."""
        import torch
        import torch.nn.functional as F

        if tensor is None:
            return tensor
        volume = self._inv_permute(torch.as_tensor(tensor).to(torch.float32))
        target_shape = (
            tuple(int(v) for v in self.origin_shape[1:])
            if self.origin_shape is not None
            else tuple(int(v) for v in volume.shape)
        )
        if tuple(volume.shape) != target_shape:
            volume = F.interpolate(
                volume.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode="nearest",
            )[0, 0]
        return volume.permute(*self.PERMUTE).round().to(dtype=torch.int16)

    def _safe_affine(self):
        spacing = tuple(
            float(value) for value in (self.img1_spacing or (1.0, 1.0, 1.0))
        )
        return safe_affine(getattr(self, "origin_meta", {}), spacing)

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
