"""UNetCNX with former UI CAM semantics; inherit model/input settings unchanged."""

_base_ = ["unetcnx.py"]
cam_protocol = dict(
    target_region="tile_prediction",
    reduction="sum",
    relu_stage="after_fusion",
)
