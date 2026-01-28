_base_ = ["../base.py"]

model = dict(
    type="SwinUNETR",
    img_size=(128, 128, 128),
    in_channels=1,
    out_channels=2,
    feature_size=48,
    use_checkpoint=True,
)  # 模型
ckpt = "checkpoint/swin_unetr_2025.pth"  # 權重檔
# ckpt = "checkpoint/swim_unetr_60_20_20.pth"  # 權重檔
default_layer = "dec4"  # 預設 CAM 層
