_base_ = ["../base.py"]

size = 96
stride = 80

model = dict(
    type="UNETR",
    in_channels=1,
    out_channels=8,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)
ckpt = "checkpoint/12_4_4_fold1/unetr.pth"  # 權重檔
default_layer = "decoder3"  # 預設 CAM 層
