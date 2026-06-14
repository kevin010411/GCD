_base_ = ["../base.py"]

model = dict(
    type="WITNET",
    # wave_level=None,
    kernel_size=7,
    patch_size=2,
    out_channels=4,
    skip_encoder_name="wf",
    deep_sup=True,
)  # 模型
ckpt = "checkpoint/inceptionWT_resca.pth"  # 權重檔
default_layer = "decoder1"  # 預設 CAM 層
