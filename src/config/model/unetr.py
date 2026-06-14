_base_ = ["../base.py"]

model = dict(
    type="UNETR",
    in_channels=1,
    out_channels=4,
    img_size=(128, 128, 128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)  # 模型
ckpt = "checkpoint/60_20_20_fold3/unetr.pth"  # 權重檔
default_layer = "decoder3"  # 預設 CAM 層
