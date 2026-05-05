_base_ = ["../base.py"]

model = dict(
    type="UNetIRC",
    # wave_level=None,
    kernel_size=5,
    patch_size=2,
    out_channels=4,
    skip_encoder_name="wf",
)  # 模型
ckpt = "checkpoint/inception_resblock.pth"  # 權重檔
default_layer = "decoder4"  # 預設 CAM 層
