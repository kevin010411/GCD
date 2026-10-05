_base_ = ["../../../config/model/unet_3d.py"]

# Reuse the UI model/checkpoint and the registered Score-CAM implementation.
inference = dict(device="cuda")

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
