"""Current default: full-volume target, mean logits, tile CAM ReLU."""
cam_protocol = dict(target_region="full_prediction", reduction="mean", relu_stage="per_tile")
