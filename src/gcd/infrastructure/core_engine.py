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
)

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
        self.origin_img = None
        self.origin_meta = {}
        self.origin_shape = None
        self.img1_spacing = self.SPACING
        self.display_metadata = self._default_display_metadata()
        self.layers = {"layer1": 1}
        self.file_name = ""
        self.patch: list[dict[str, object]] = []
        self.target_class = 1
        self.xai_cache_key = ""
        self.cam_methods: dict[str, CamMethod] = {
            GradCamMethod.id: GradCamMethod(self._gradcam_objective),
            GradCAMTestMethod.id: GradCAMTestMethod(),
            SaliencyMapMethod.id: SaliencyMapMethod(self._gradcam_objective),
            PerturbationOcclusionMethod.id: PerturbationOcclusionMethod(),
        }
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

    def set_target_class(self, target_class: int) -> None:
        self.target_class = int(target_class)

    def available_cam_methods(self, category: str | None = None) -> list[dict[str, object]]:
        methods = self.cam_methods.values()
        if category is not None:
            methods = [method for method in methods if getattr(method, "category", "") == category]
        return [
            {
                "id": method.id,
                "name": method.display_name,
                "uses_layer_controls": bool(method.uses_layer_controls),
            }
            for method in methods
        ]

    def default_feature_size(self) -> int:
        return list(self.layers.values())[0]

    def dataset_input(self) -> DatasetInput:
        return DatasetInput(
            img0=deepcopy(self.img0),
            img1=deepcopy(self.img1),
            origin_img=deepcopy(self.origin_img),
            origin_meta=deepcopy(self.origin_meta),
            origin_shape=deepcopy(self.origin_shape),
            img1_spacing=deepcopy(self.img1_spacing),
            display_metadata=deepcopy(self.display_metadata),
            layers=deepcopy(self.layers),
            file_name=self.file_name,
            target_class=self.target_class,
            active_method_id=self.active_method_id,
            xai_cache_key=self.xai_cache_key,
        )

    def load_dataset_input(self, dataset_input: DatasetInput) -> None:
        self.cam = None
        self.volume_data = None
        self.img0 = deepcopy(dataset_input.img0)
        self.img1 = deepcopy(dataset_input.img1)
        self.origin_img = deepcopy(dataset_input.origin_img)
        self.origin_meta = deepcopy(dataset_input.origin_meta)
        self.origin_shape = deepcopy(dataset_input.origin_shape)
        self.img1_spacing = deepcopy(dataset_input.img1_spacing)
        self.display_metadata = deepcopy(dataset_input.display_metadata)
        self.layers = deepcopy(dataset_input.layers)
        self.file_name = str(dataset_input.file_name)
        self.patch = []
        self.target_class = int(dataset_input.target_class)
        self.active_method_id = str(dataset_input.active_method_id)
        self.model_output = None
        self.xai_cache_key = str(dataset_input.xai_cache_key)

    def _resolve_cam_method(self, method: str | None) -> CamMethod:
        requested = (method or self.active_method_id or GradCamMethod.id).strip().lower()
        if requested in self.cam_methods:
            return self.cam_methods[requested]
        self._log(f"未知 CAM method '{method}'，改用預設方法: {GradCamMethod.id}")
        return self.cam_methods[GradCamMethod.id]


    @staticmethod
    def _gradcam_objective(logits: torch.Tensor, target_class: int) -> torch.Tensor:
        import torch

        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        index = torch.argmax(logits[0], dim=0)
        loss = (logits[0, target_class] * (index == target_class)).sum()
        return loss

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

        with _timer("資料前處理"):
            try:
                self.origin_meta = dict(self.img0.meta)
            except Exception:
                self.origin_meta = {}
            self.origin_shape = tuple(self.img0.shape)

            self.img1 = mt.Spacing(mode="bilinear", pixdim=self.SPACING)(self.img0)
            width, depth = self.SIZE + self.STRIDE, self.SIZE
            self.img1 = mt.SpatialPad(
                spatial_size=(width, width, depth), mode="constant", value=0
            )(self.img1)
            img1_affine = self._extract_affine(self.img1)

            width, depth = self.SIZE + self.STRIDE, self.SIZE
            shape = list(self.img1.shape)
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
                image[tuple(slices)] = self.img1
                self.img1 = image
                messages.append("info: image is zero padded")
                img1_affine = self._shift_affine_for_padding(img1_affine, pad_offsets)

            self.img1 = mt.ScaleIntensityRange(
                a_min=-42, a_max=423, b_min=0, b_max=1, clip=True
            )(self.img1)
            self.img1_spacing = (
                self.SPACING[self.PERMUTE[0]],
                self.SPACING[self.PERMUTE[1]],
                self.SPACING[self.PERMUTE[2]],
            )
            self.display_metadata = self._build_display_metadata(img1_affine)
            self.volume_data = self.img1[0].permute(*self.PERMUTE).to(torch.float32)
            self.cam = torch.zeros_like(self.volume_data)
            self.model_output = None
            self.patch = []
            self.layers = {self.cfg["default_layer"]: 1}
            self.xai_cache_key = ""

        for message in messages:
            self._log(message)
        return messages

    def prepare_xai_inputs(self, method: str | None = None) -> None:
        import torch

        if not self.file_name or self.img1 is None:
            raise ValueError("尚未載入檔案，無法準備 XAI 輸入。")
        cam_method = self._resolve_cam_method(method)
        self.active_method_id = cam_method.id
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
            for x, y in tiles:
                model.zero_grad(set_to_none=True)
                if img2.grad is not None:
                    img2.grad = None
                tile_input = img2[
                    ...,
                    x0 + x : x0 + x + self.SIZE,
                    y0 + y : y0 + y + self.SIZE,
                    z0 : z0 + self.SIZE,
                ]
                tile_input.retain_grad()
                logits = model(tile_input)

                layers_by_name = (
                    model.layers
                    if hasattr(model, "layers") and model.layers
                    else {}
                )

                if cam_method.uses_layer_controls and not layers_by_name:
                    raise RuntimeError(
                        "model.layers 未填入。請確認模型 forward 在 requires_grad=True 時"
                        "會保留中間層與梯度。"
                    )

                self.patch.append(
                    cam_method.collect_patch_data(
                        CamPatchContext(
                            input_tensor=tile_input,
                            logits=logits,
                            layers_by_name=layers_by_name,
                            target_class=self.target_class,
                            objective=self._gradcam_objective,
                        )
                    )
                )

            self.layers = (
                {key: value.size(1) for key, value in model.layers.items()}
                if cam_method.uses_layer_controls
                else {"input": 1}
            )
            self.model_output = logits.detach().to("cpu")

            del model, img2, pth, sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cfg_name = str(getattr(self.cfg, "filename", "") or "")
        self.xai_cache_key = f"{cfg_name}|{self.target_class}|{cam_method.id}"

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
                    selected_layer,
                    n1,
                    n2,
                    (self.SIZE, self.SIZE, self.SIZE),
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

            cam = torch.maximum(cam, torch.tensor(0))
            cam -= torch.min(cam)
            maximum = torch.max(cam)
            if maximum > 0:
                cam /= maximum

            self.cam = cam.permute(*self.PERMUTE)
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
