_base_ = ["../base.py"]

model = dict(
    type="INCEPTION_RESBLOCK",
    # wave_level=None,
    kernel_size=5,
    patch_size=2,
    out_channels=4,
    skip_encoder_name="wf",
    deep_sup=True,
    res_block=False,
)  # 模型
ckpt = "checkpoint/inception_resblock.pth"  # 權重檔
default_layer = "dec1"  # 預設 CAM 層
