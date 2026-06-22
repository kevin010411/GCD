import unittest

import torch

from src.gcd.infrastructure.xai.methods.cam_methods import PerturbationOcclusionMethod
from src.gcd.infrastructure.xai.tiling.tile_strategy import SlidingWindowTileStrategy
from src.gcd.infrastructure.xai.runners.xai_cam_runner import XaiCamRunRequest, XaiCamRunner


class XaiCamRunnerTests(unittest.TestCase):
    def test_sliding_window_averages_overlapping_tile_contributions(self) -> None:
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(3, 2, 2),
            patch_size=2,
            stride=1,
        )
        pred = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
        patches = [
            {
                "method": "perturb_occlusion",
                "pred": pred,
                "importance_map": torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
            },
            {
                "method": "perturb_occlusion",
                "pred": pred,
                "importance_map": torch.full((1, 1, 2, 2, 2), 3.0, dtype=torch.float32),
            },
        ]

        result = XaiCamRunner().run(
            XaiCamRunRequest(
                method=PerturbationOcclusionMethod(),
                patches=patches,
                layers={"input": 1},
                img1=torch.ones((1, 3, 2, 2), dtype=torch.float32),
                size=2,
                stride=1,
                permute=(0, 1, 2),
                default_layer="input",
                tile_plan=plan,
            )
        )

        self.assertAlmostEqual(float(result.cam[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(result.cam[1, 0, 0]), 0.5)
        self.assertAlmostEqual(float(result.cam[2, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
