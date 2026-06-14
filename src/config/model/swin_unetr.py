_base_ = ["../base.py"]

model = dict(
    type="SwinUNETR",
    img_size=(128, 128, 128),
    in_channels=1,
    out_channels=4,
    feature_size=48,
)  # 模型
ckpt = "checkpoint/60_20_20_fold3/swin_unetr.pth"  # 權重檔
# ckpt = "checkpoint/swim_unetr_60_20_20.pth"  # 權重檔
default_layer = "encoder10"  # 預設 CAM 層
