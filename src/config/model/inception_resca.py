_base_ = ["../base.py"]

model = dict(
    type="INCEPTION_RESCA",
    # wave_level=None,
    kernel_size=5,
    patch_size=2,
    out_channels=4,
    skip_encoder_name="wf",
    deep_sup=True,
    res_block=True,
)  # 模型
ckpt = "checkpoint/inception_resca.pth"  # 權重檔
default_layer = "dec1"  # 預設 CAM 層
