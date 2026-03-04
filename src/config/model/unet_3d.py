_base_ = ["../base.py"]

model = dict(
    type="UNet",
    act="RELU",
    norm="BATCH",
    out_channels=4,
)  # 模型
# ckpt = "checkpoint/3d_unet_2025.pth"  # 權重檔
ckpt = "checkpoint/60_20_20_fold3/unet3d.pth"  # 權重檔
default_layer = "decoder 1"  # 預設 CAM 層
