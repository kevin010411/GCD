_base_ = ["../../src/config/model/unet_3d.py"]

# Reuse the UI model/checkpoint and the registered Score-CAM implementation.
inference = dict(
    device="cuda",
    roi_size=(128, 128, 128),
    sw_batch_size=1,
    overlap=0.25,
    blend_mode="constant",
    warmup_runs=0,
    benchmark_runs=1,
)
preprocessing = dict(
    spacing=(0.7, 0.7, 1.0),
    intensity_input_range=(-42.0, 423.0),
    intensity_output_range=(0.0, 1.0),
)
metrics = dict(include_background=False, empty_score=1.0)

XaiMethods = [
    dict(
        type="RegistryXaiMethod",
        id="scorecam",
        method="scorecam",
        layer="decoder 2",
        target_class=[1],
        params=dict(_ui_tiled=True),
    ),
]
XaiMetrics = []
XaiAnswer = None
