import os, sys
import torch, torch.nn.functional as F
import monai.transforms as mt
from mmengine import Config

from .utils import build_model, timer


class gcd_core:

    def __init__(self, cfg_path: str, save_dir: str | None = None):

        self.cfg = Config.fromfile(cfg_path)

        # 這些屬性會被 compute_cam 等方法用到
        self.SIZE = self.cfg["size"]
        self.STRIDE = self.cfg["stride"]
        self.SPACING = self.cfg["spacing"]
        self.PERMUTE = self.cfg["permute"]

        # 初始狀態
        self.cam = torch.zeros([256, 256, 150], dtype=torch.float32)
        self.volume_data = torch.zeros([256, 256, 150], dtype=torch.float32)
        self.img0 = None
        self.img1 = None
        self.img1_spacing = self.SPACING
        self.layers = {"layer1": 1}
        self.file_name = ""
        self.patch = []

        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def set_config(self, config_path):
        self.cfg = Config.fromfile(config_path)
        print(f"已設定Config為:{self.cfg}")

    def load_and_process_input(self, input_file):
        self.file_name = input_file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.origin_img = mt.LoadImage()(input_file)
        self.img0 = mt.EnsureChannelFirst()(self.origin_img)

        with timer("資料前處理"):
            # --- 記錄原始 metadata 與 shape 以供存檔還原 ---
            try:
                # MetaTensor
                self.origin_meta = dict(self.img0.meta)
            except Exception:
                # 後備：某些情況 LoadImage 可能回傳 numpy，則沒有 meta
                self.origin_meta = {}
            self.origin_shape = tuple(self.img0.shape)

            self.img1 = mt.Spacing(mode="bilinear", pixdim=self.SPACING)(self.img0)

            # zero pad image if too小
            W, D = self.SIZE + self.STRIDE, self.SIZE
            self.img1 = mt.SpatialPad(spatial_size=(W, W, D), mode="constant", value=0)(
                self.img1
            )
            W, D = self.SIZE + self.STRIDE, self.SIZE
            shape = list(self.img1.shape)
            slices = [slice(None), slice(None), slice(None), slice(None)]
            if shape[1] < W:
                x = (W - shape[1]) // 2
                slices[1] = slice(x, x + shape[1])
                shape[1] = W
            if shape[2] < W:
                x = (W - shape[2]) // 2
                slices[2] = slice(x, x + shape[2])
                shape[2] = W
            if shape[3] < D:
                x = (D - shape[3]) // 2
                slices[3] = slice(x, x + shape[3])
                shape[3] = D
            if any(s.start is not None for s in slices):
                img = torch.zeros(shape)
                img[tuple(slices)] = self.img1
                self.img1 = img
                print("info: image is zero padded")

            # intensity normalize
            self.img1 = mt.ScaleIntensityRange(
                a_min=-42, a_max=423, b_min=0, b_max=1, clip=True
            )(self.img1)
            # 記錄重採樣 spacing（配合可視化軸向）
            self.img1_spacing = (
                self.SPACING[self.PERMUTE[0]],
                self.SPACING[self.PERMUTE[1]],
                self.SPACING[self.PERMUTE[2]],
            )

        with timer("載入模型"):
            # 載入模型
            model = build_model(self.cfg.model).to(device)
            ckpt_path = self.cfg.ckpt
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            pth = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in pth:
                sd = pth["state_dict"].copy()
            else:
                sd = pth.copy()
            # for key in list(sd.keys()):
            #     if "ds" in key:
            #         sd.pop(key)
            # if self.cfg.model.type == "UNETR":
            #     sd.pop("vit.patch_embedding.position_embeddings", None)

            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(
                    f"warn: missing keys: {len(missing)} (showing first 5) {missing[:5]}"
                )
            if unexpected:
                print(
                    f"warn: unexpected keys: {len(unexpected)} (showing first 5) {unexpected[:5]}"
                )
            model.eval()

        # 生成帶梯度的輸入
        img2 = self.img1.unsqueeze(0).to(device)  # [B=1, C=1, X, Y, Z]
        img2.requires_grad_()

        with timer("模型推論", track_gpu=True):
            print("info: computing, this may take a while ", end="", flush=True)
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
                logits = model(
                    img2[
                        ...,
                        x0 + x : x0 + x + self.SIZE,
                        y0 + y : y0 + y + self.SIZE,
                        z0 : z0 + self.SIZE,
                    ]
                )

                index = torch.argmax(logits[0], dim=0)
                loss = (logits[0, 1] * (index == 1)).sum()
                loss.backward()

                if not hasattr(model, "layers") or not model.layers:
                    raise RuntimeError(
                        "model.layers 未填入。請確認你的模型 forward 在 requires_grad=True 時，"
                        "會把中間層輸出存進 model.layers 並 retain_grad()。"
                    )

                # 儲存每一塊的 (activation * grad)
                self.patch.append(
                    {
                        k: (v.detach() * v.grad.detach()).cpu()
                        for k, v in model.layers.items()
                    }
                )

                self.patch[-1]["pred"] = logits.to("cpu")
                print(".", end="", flush=True)

            print(" done")

            # 記錄每個層的 channel 數量，供 compute_cam() 限縮 n2
            self.layers = {k: v.size(1) for k, v in model.layers.items()}

            # 釋放
            del model, img2, pth, sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def compute_cam(self, layer=None, n1=0, n2=999, use_overlay=True):
        if not layer:
            # 若未指定，就依當前 config 的預設層
            layer = self.cfg["default_layer"]
        if layer not in self.layers:
            raise KeyError(f"layer '{layer}' 不存在，可選：{list(self.layers.keys())}")
        with timer(f"{layer=}計算GradCAM"):

            n2 = min(n2, self.layers[layer])

            shape = list(self.img1[0].shape)
            cam = torch.zeros(shape, dtype=torch.float32)

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

            for n, (x, y) in enumerate(tiles):
                q = torch.sum(self.patch[n][layer][:, n1:n2, ...], dim=1).unsqueeze(0)
                q = F.interpolate(
                    q, size=(self.SIZE, self.SIZE, self.SIZE), mode="trilinear"
                )

                # --- 模型輸出重建（class-1 機率）---
                p1 = self.patch[n]["pred"]
                p1 = F.interpolate(
                    p1, size=(self.SIZE, self.SIZE, self.SIZE), mode="trilinear"
                )

                # 平滑重疊區
                p = self.SIZE - self.STRIDE  # overlap
                if n == 0 or n == 1:
                    for i in range(p):
                        w = (p - i) / p
                        q[0, 0, self.STRIDE + i, :, :] *= w
                        p1[0, 0, self.STRIDE + i, :, :] *= w
                if n == 2 or n == 3:
                    for i in range(p):
                        w = i / p
                        q[0, 0, i, :, :] *= w
                        p1[0, 0, i, :, :] *= w
                if n == 0 or n == 2:
                    for i in range(p):
                        w = (p - i) / p
                        q[0, 0, :, self.STRIDE + i, :] *= w
                        p1[0, 0, :, self.STRIDE + i, :] *= w
                if n == 1 or n == 3:
                    for i in range(p):
                        w = i / p
                        q[0, 0, :, i, :] *= w
                        p1[0, 0, :, i, :] *= w

                xs = slice(x0 + x, x0 + x + self.SIZE)
                ys = slice(y0 + y, y0 + y + self.SIZE)
                zs = slice(z0, z0 + self.SIZE)

                cam[xs, ys, zs] += q[0, 0]
                model_out[:, :, xs, ys, zs] += p1

            cam = torch.maximum(cam, torch.tensor(0))
            cam -= torch.min(cam)
            m = torch.max(cam)
            # print(f"{m.item():.3f}", end=" ", flush=True)
            if m > 0:
                cam /= m

            self.cam = ((cam + 1) * 400).permute(*self.PERMUTE)  # 400~800
            if use_overlay:
                self.volume_data = (self.img1[0] * 300).permute(*self.PERMUTE)  # 0~300
            else:
                self.volume_data = None  # 0~300

            self.model_output = torch.argmax(model_out, dim=1)[0].permute(*self.PERMUTE)
            # print(f"max:{self.cam.max()},min:{self.cam.min()}")

        with timer("儲存GradCAM"):
            # ---- 是否儲存 ----
            _dir = self.save_dir
            if _dir:
                # breakpoint()  # 輸出的內容有很多個class
                os.makedirs(_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(self.file_name))[0]
                self._save_volume(
                    cam.permute(*self.PERMUTE),
                    os.path.join(_dir, f"{base}_{self.cfg.model.type}_{layer}_cam"),
                )
                self._save_volume(
                    self.model_output,
                    os.path.join(_dir, f"{base}_{self.cfg.model.type}_pred"),
                )
                self._save_volume(self.volume_data, os.path.join(_dir, f"{base}_img"))
                print(f"\nsaved to: {_dir}")

    def _inv_permute(self, t: torch.Tensor) -> torch.Tensor:
        """將 (X, Y, Z) with self.PERMUTE 還原回與 self.img0 相同的空間軸順序。"""
        if t is None:
            return t
        inv = [0, 0, 0]
        for i, p in enumerate(self.PERMUTE):
            inv[p] = i
        return t.permute(*inv)

    def _resize(self, v: torch.Tensor, size):
        # 離散→nearest；連續→trilinear；先轉 float32 再插值
        mode = (
            "trilinear"
            if v.dtype in (torch.float32, torch.float16, torch.float64)
            else "nearest"
        )
        v = v.to(torch.float32)
        x = v.unsqueeze(0).unsqueeze(0)
        if mode == "trilinear":
            x = F.interpolate(x, size=size, mode=mode, align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode=mode)  # 不要傳 align_corners！
        return x[0, 0]

    def _to_origin_space(self, t: torch.Tensor) -> torch.Tensor:
        """把 t (X,Y,Z after PERMUTE) 轉回 self.img0 的空間大小與順序。"""
        if t is None:
            return t
        # 還原軸順序
        v = self._inv_permute(t)
        want = (
            list(self.origin_shape[1:])
            if self.origin_shape is not None
            else list(v.shape)
        )
        if tuple(v.shape) != tuple(want):
            v = self._resize(v, want)
        else:
            v = v.to(torch.float32)
        return v

    def _safe_affine(self):
        """Pick an affine: prefer origin_meta['affine']; else diagonal from spacing; else I."""
        import numpy as np

        # default spacing tuple
        spacing = None
        if (
            self.origin_meta
            and "pixdim" in self.origin_meta
            and len(self.origin_meta["pixdim"]) >= 4
        ):
            # MONAI sometimes stores NIfTI pixdim; take 1..3
            try:
                spacing = tuple(map(float, self.origin_meta["pixdim"][1:4]))
            except Exception:
                spacing = None
        if spacing is None:
            spacing = tuple(
                float(x)
                for x in (
                    self.img1_spacing
                    if hasattr(self, "img1_spacing") and self.img1_spacing
                    else (1.0, 1.0, 1.0)
                )
            )

        # choose affine
        A = None
        if self.origin_meta and "affine" in self.origin_meta:
            A = self.origin_meta["affine"]
        if A is None:
            A = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype("float32")
        return A

    def _save_volume(self, t: torch.Tensor, stem: str, exist_ok=True):
        """儲存張量為 .pt，若可用 nibabel 則另存為 .nii.gz（帶 spacing）"""
        # 1) 存 .pt
        # torch.save(t.cpu(), f"{stem}.pt")
        # 2) 盡量存 NIfTI
        if t is None:
            return
        try:
            import numpy as np
            import nibabel as nib

            arr = self._to_origin_space(t).detach().cpu().numpy().astype("float32")
            affine = None
            if self.origin_meta and "affine" in self.origin_meta:
                affine = self.origin_meta["affine"]
            if affine is None:
                sx, sy, sz = self.img1_spacing
            affine = self._safe_affine()
            nii = nib.Nifti1Image(arr, affine=affine)
            if exist_ok:
                nib.save(nii, f"{stem}.nii.gz")
            elif os.path.exists(f"{stem}.nii.gz"):
                raise FileExistsError(
                    f"File({f'{stem}.nii.gz'}) already exists, use exist_ok=True to overwrite."
                )

        except Exception as e:
            print(f"note: nibabel not available or failed ({e});")


if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "dat/demo.1.nii.gz"
    layer = (
        sys.argv[2] if len(sys.argv) > 2 else None
    )  # 若 None，走各自 config 的 default_layer
    outfile = sys.argv[3] if len(sys.argv) > 3 else "cam"
    cfgname = (
        sys.argv[4] if len(sys.argv) > 4 else "cnx"
    )  # 第四個參數：選 "cnx" 或 "unet"

    # 匯出檔時若想維持原圖座標，可改成 (0,1,2)
    PERMUTE = (0, 1, 2)

    core = gcd_core(cfgname)
    # 儲存與顯示的 permute 可獨立於推論時使用的 permute
    core.PERMUTE = PERMUTE

    core.load_and_process_input(infile)
    core.compute_cam(layer)

    # 尺度對齊原圖再存
    cam = F.interpolate(
        core.cam.unsqueeze(0).unsqueeze(0), size=list(core.img0.shape), mode="trilinear"
    )
    mt.SaveImage()(cam[0, 0], core.img0.meta, outfile)
