"""Shared GCD transforms and sliding-window inference settings."""

_base_ = ["../cam/benchmark.py"]

preprocessing = dict(
    channel_dim=None,
    force_3d=False,
    spacing=(0.7, 0.7, 1.0),
    intensity_input_range=(-42.0, 423.0),
    intensity_output_range=(0.0, 1.0),
    steps=[
        dict(type="Spacing", mode="bilinear", label_mode="nearest"),
        dict(
            type="SpatialPad",
            spatial_size="window",
            method="symmetric",
            mode="constant",
        ),
        dict(type="ScaleIntensityRange", clip=True),
        dict(type="EnsureType", dtype="float32"),
    ],
)
inference = dict(
    tile_strategy="sliding_window",
    device="auto",
    roi_size=(128, 128, 128),
    sw_batch_size=1,
    overlap=0.25,
    blend_mode="gaussian",
    sigma_scale=0.125,
    padding_mode="constant",
    cval=0.0,
    warmup_runs=0,
    benchmark_runs=1,
)
postprocessing = dict(keep_largest_connected_component=False)
display = dict(permute=(1, 2, 0))
metrics = dict(include_background=False, empty_score=1.0)
