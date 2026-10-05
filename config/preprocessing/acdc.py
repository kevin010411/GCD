"""xai_hw ACDC transform order, including padding before normalization."""

_base_ = ["gcd.py"]
preprocessing = dict(
    _delete_=True,
    channel_dim="no_channel",
    force_3d=True,
    spacing=(1.5, 1.5, 5.0),
    steps=[
        dict(type="Spacing", mode="bilinear", label_mode="nearest"),
        dict(type="Orientation", axcodes="RAS"),
        dict(
            type="SpatialPad",
            spatial_size="window",
            method="symmetric",
            mode="constant",
        ),
        dict(type="NormalizeIntensity", nonzero=True, channel_wise=True),
        dict(type="EnsureType", dtype="float32"),
    ],
)
inference = dict(
    roi_size=(96, 96, 32),
    sw_batch_size=4,
    blend_mode="gaussian",
    tile_strategy="sliding_window",
)
postprocessing = dict(keep_largest_connected_component=True)
metrics = dict(empty_score=0.0)
