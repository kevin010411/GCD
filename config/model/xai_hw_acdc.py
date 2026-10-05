"""Model and inference settings for the xai_hw ACDC parity protocol.

The GCD UI and both experiment runners use this model/preprocessing config.
Override ckpt or pass --checkpoint when using a different filesystem layout.
"""

_base_ = ["../preprocessing/acdc.py"]

model = dict(
    type="MonaiUNet",
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    channels=(64, 128, 256, 256),
    strides=(2, 2, 2),
    num_res_units=3,
    act="RELU",
    norm="BATCH",
    dropout=0.0,
    bias=True,
    adn_ordering="NDA",
)
# Shared-container default; override ckpt for other filesystem layouts.
ckpt = "D:/KevinFu/GCD/checkpoint/acdc_60_20_20_fold1/unet3d.pth"
benchmark = dict(feature_layers="last-three-conv")
XaiMethods = []
XaiMetrics = []
XaiAnswer = None
default_layer = "model.1.submodule.2.1.conv.unit0.conv"
