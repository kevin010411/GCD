_base_ = ["../base.py"]

model = dict(
    type="Cotr",
    norm_cfg="IN",
    activation_cfg="LeakyReLU",
    img_size=(128, 128, 128),
    num_classes=2,
    weight_std=False,
    deep_supervision=False,
)  # 模型
ckpt = "checkpoint/cotr_2025.pth"  # 權重檔
default_layer = "decoder0"  # 預設 CAM 層
