_base_ = ["../base.py"]

model = dict(
    type="UNETR",
    in_channels=1,
    out_channels=2,
    img_size=(128, 128, 128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)  # 模型
ckpt = "checkpoint/unetr_2025.pth"  # 權重檔
default_layer = "dec1"  # 預設 CAM 層
