import unittest
from unittest import mock

import torch

from src.gcd.infrastructure.xai.methods.cam_methods import (
    PerturbationOcclusionMethod,
    ScoreCamMethod,
)
from src.gcd.infrastructure.xai.tiling.tile_strategy import (
    SlidingWindowTileStrategy,
)
from src.gcd.infrastructure.xai.runners.xai_cam_runner import (
    XaiCamRunRequest,
    XaiCamRunner,
)
from src.gcd.infrastructure.xai.tiling.scorecam_blend import gaussian_importance_map


class XaiCamRunnerTests(unittest.TestCase):
    def test_scorecam_uses_gaussian_overlap_without_cropping_outer_edges(self) -> None:
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(4, 3, 3), patch_size=3, stride=1
        )
        patches = []
        for value in (1.0, 3.0):
            patches.append({
                "method": "scorecam",
                "pred": torch.zeros((1, 2, 1, 1, 1)),
                "selected_layer": "feature",
                "feature_start": 0,
                "feature_stop": 1,
                "scores": torch.zeros(1),
                "layers": {"feature": {"activation": torch.full((1, 1, 3, 3, 3), value)}},
            })
        result = XaiCamRunner().run(XaiCamRunRequest(
            method=ScoreCamMethod(lambda logits, target: logits[0, target].sum()),
            patches=patches,
            layers={"feature": 1},
            img1=torch.ones((1, 4, 3, 3)),
            size=3,
            stride=1,
            permute=(0, 1, 2),
            default_layer="feature",
            tile_plan=plan,
            method_params={"_blend_mode": "gaussian"},
        ))

        axis_weight = gaussian_importance_map((3, 3, 3))[:, 1, 1]
        raw = torch.tensor([
            1.0,
            (axis_weight[1] + 3 * axis_weight[0]) / (axis_weight[1] + axis_weight[0]),
            (axis_weight[2] + 3 * axis_weight[1]) / (axis_weight[2] + axis_weight[1]),
            3.0,
        ])
        self.assertTrue(torch.allclose(result.cam[:, 1, 1], (raw - 1) / 2))

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
        self.assertEqual(result.model_output.dtype, torch.uint8)

    def test_sliding_strategy_allocates_coverage_volume(self) -> None:
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(4, 4, 2),
            patch_size=2,
            stride=1,
        )
        pred = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
        patches = [
            {
                "method": "perturb_occlusion",
                "pred": pred.clone(),
                "importance_map": torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
            }
            for _ in plan.regions
        ]

        with mock.patch("torch.zeros", wraps=torch.zeros) as zeroes:
            XaiCamRunner().run(
                XaiCamRunRequest(
                    method=PerturbationOcclusionMethod(),
                    patches=patches,
                    layers={"input": 1},
                    img1=torch.ones((1, 4, 4, 2), dtype=torch.float32),
                    size=2,
                    stride=1,
                    permute=(0, 1, 2),
                    default_layer="input",
                    tile_plan=plan,
                )
            )

        volume_shape = [4, 4, 2]
        full_volume_allocations = [
            call
            for call in zeroes.call_args_list
            if call.args and list(call.args[0]) == volume_shape
        ]
        self.assertEqual(len(full_volume_allocations), 2)

    def test_prediction_uses_int16_when_class_ids_exceed_uint8(self) -> None:
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(3, 2, 2),
            patch_size=2,
            stride=1,
        )
        pred = torch.zeros((1, 300, 1, 1, 1), dtype=torch.float32)
        pred[:, 299] = 1
        patches = [
            {
                "method": "perturb_occlusion",
                "pred": pred,
                "importance_map": torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
            }
            for _ in plan.regions
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

        self.assertEqual(result.model_output.dtype, torch.int16)
        self.assertTrue(
            torch.equal(
                result.model_output,
                torch.full((3, 2, 2), 299, dtype=torch.int16),
            )
        )


if __name__ == "__main__":
    unittest.main()
