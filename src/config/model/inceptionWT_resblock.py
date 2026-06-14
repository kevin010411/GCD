_base_ = ["../base.py"]

model = dict(
    type="INCEPTIONWT_RESBLOCK",
    # wave_level=None,
    kernel_size=7,
    patch_size=2,
    out_channels=4,
    skip_encoder_name="wf",
    wave_level=2,
    deep_sup=True,
)  # 模型
ckpt = "checkpoint/inceptionWT_resblock.pth"  # 權重檔
default_layer = "decoder1"  # 預設 CAM 層
