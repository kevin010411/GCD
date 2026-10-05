"""UNet3D with former UI CAM semantics; inherit model/input settings unchanged."""

_base_ = ["unet_3d.py"]
cam_protocol = dict(
    target_region="tile_prediction",
    reduction="sum",
    relu_stage="after_fusion",
)
