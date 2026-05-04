_base_ = ["../base.py"]

model = dict(
    type="UNetIRC", skip_encoder_name="CBAM", patch_size=2, out_channels=4
)  # 模型
ckpt = "checkpoint/60_20_20_fold1/unetirc_11x5.pth"  # 權重檔
default_layer = "dec4"  # 預設 CAM 層
