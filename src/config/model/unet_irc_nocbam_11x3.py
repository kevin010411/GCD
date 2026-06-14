_base_ = ["../base.py"]

model = dict(
    type="UNetIRC",
    skip_encoder_name="CBAM",
    patch_size=2,
    out_channels=4,
    hwd_kernel_size=3,
    use_cbam=False,
)  # 模型
ckpt = "checkpoint/60_20_20_fold3/unetirc_nocbam11x3.pth"  # 權重檔
default_layer = "decoder4"  # 預設 CAM 層
