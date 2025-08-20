import os, sys
import importlib
import torch, torch.nn.functional as F
import monai.transforms as mt

# -------------------------
# 可切換的模型設定
# -------------------------
CONFIGS = {
    # ConvNeXt 版（你的 UNETCNX_A1）
    "cnx": {
        "import": "dat.unetcnx:UNETCNX_A1",  # 模型類別位置
        "ckpt": "dat/unetcnx_2025.pth",  # 權重檔
        "size": 128,  # patch size (x=y=z)
        "stride": 112,  # 滑窗 stride
        "spacing": (0.7, 0.7, 1.0),  # 重採樣 spacing
        "permute": (1, 2, 0),  # 顯示時軸交換
        "default_layer": "dec4",  # 預設 CAM 層
    },
    # MONAI UNet 版（請確保你的 dat.unet.UNet 的 forward 會把最上層 down/up 拆到 self.layers）
    "unet": {
        "import": "dat.unet:UNet",
        "ckpt": "dat/3d_unet_2025.pth",  # 建議給 UNet 自己的權重檔
        "size": 128,
        "stride": 112,
        "spacing": (0.7, 0.7, 1.0),
        "permute": (1, 2, 0),
        "act": "RELU",
        "norm": "BATCH",
        "default_layer": "decoder 1",  # 預設 CAM 層
    },
}


def _import_class(path: str):
    """'pkg.mod:ClassName' -> 類別物件"""
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)


class gcd_core:
    def __init__(self, cfg_name: str = "cnx"):
        if cfg_name not in CONFIGS:
            raise ValueError(f"Unknown cfg '{cfg_name}', choices = {list(CONFIGS)}")
        self.cfg = CONFIGS[cfg_name]

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

    def load_and_process_input(self, input_file):
        self.file_name = input_file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img0 = mt.LoadImage()(input_file).flip(0)
        self.img1 = mt.Spacing(mode="bilinear", pixdim=self.SPACING)(
            self.img0.unsqueeze(0)
        )

        # zero pad image if too小
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

        # 載入模型
        ModelCls = _import_class(self.cfg["import"])
        ckpt_path = self.cfg["ckpt"]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        pth = torch.load(ckpt_path, map_location="cpu")
        # 移除訓練用的深監督頭（如有）
        if "state_dict" in pth:
            sd = pth["state_dict"].copy()
        else:
            sd = pth.copy()
        for key in list(sd.keys()):
            if "ds" in key:
                sd.pop(key)

        model = ModelCls().to(device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"warn: missing keys: {len(missing)} (showing first 5) {missing[:5]}")
        if unexpected:
            print(
                f"warn: unexpected keys: {len(unexpected)} (showing first 5) {unexpected[:5]}"
            )
        model.eval()

        # 生成帶梯度的輸入
        img2 = self.img1.unsqueeze(0).to(device)  # [B=1, C=1, X, Y, Z]
        img2.requires_grad_()

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
        tiles = [(0, 0), (0, self.STRIDE), (self.STRIDE, 0), (self.STRIDE, self.STRIDE)]
        for x, y in tiles:
            logits = model(
                img2[
                    ...,
                    x0 + x : x0 + x + self.SIZE,
                    y0 + y : y0 + y + self.SIZE,
                    z0 : z0 + self.SIZE,
                ]
            )
            # 以 class=1 當目標（與你原本一致）
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
        n2 = min(n2, self.layers[layer])

        shape = list(self.img1[0].shape)
        cam = torch.zeros(shape, dtype=torch.float32)
        x0, y0, z0 = list(
            (
                torch.tensor(self.img1[0].shape)
                - torch.tensor(
                    [self.STRIDE + self.SIZE, self.STRIDE + self.SIZE, self.SIZE]
                )
            )
            // 2
        )

        tiles = [(0, 0), (0, self.STRIDE), (self.STRIDE, 0), (self.STRIDE, self.STRIDE)]
        for n, (x, y) in enumerate(tiles):
            q = torch.sum(self.patch[n][layer][:, n1:n2, ...], dim=1).unsqueeze(0)
            q = F.interpolate(
                q, size=(self.SIZE, self.SIZE, self.SIZE), mode="trilinear"
            )

            # 平滑重疊區
            p = self.SIZE - self.STRIDE  # overlap
            if n == 0 or n == 1:
                for i in range(p):
                    q[0, 0, self.STRIDE + i, :, :] *= (p - i) / p
            if n == 2 or n == 3:
                for i in range(p):
                    q[0, 0, i, :, :] *= i / p
            if n == 0 or n == 2:
                for i in range(p):
                    q[0, 0, :, self.STRIDE + i, :] *= (p - i) / p
            if n == 1 or n == 3:
                for i in range(p):
                    q[0, 0, :, i, :] *= i / p

            cam[
                x0 + x : x0 + x + self.SIZE,
                y0 + y : y0 + y + self.SIZE,
                z0 : z0 + self.SIZE,
            ] += q[0, 0]

        cam = torch.maximum(cam, torch.tensor(0))
        cam -= torch.min(cam)
        m = torch.max(cam)
        print(f"{m.item():.3f}", end=" ", flush=True)
        if m > 0:
            cam /= m

        # if use_overlay:
        #     cam[cam > 0.1] += 1  # 0~40 , 440~800
        #     # cam += 1  # 400~800
        #     self.cam = (cam * 400 + self.img1[0] * 300).permute(*self.PERMUTE)
        #     # self.cam = max(cam * 400, self.img1[0] * 300).permute(*self.PERMUTE)
        # else:
        #     self.cam = (cam * 900).permute(*self.PERMUTE)

        self.cam = ((cam + 1) * 400).permute(*self.PERMUTE)  # 400~800
        self.volume_data = (self.img1[0] * 300).permute(*self.PERMUTE)  # 0~300


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
