"""Same ACDC model/input as current default, with former UI CAM semantics."""
_base_ = ["xai_hw_acdc.py"]
cam_protocol = dict(target_region="tile_prediction", reduction="sum", relu_stage="after_fusion")
