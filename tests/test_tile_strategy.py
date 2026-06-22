import unittest

from src.gcd.infrastructure.xai.tiling.tile_strategy import (
    LegacyFourTileStrategy,
    SlidingWindowTileStrategy,
    TileStrategyResolver,
)


class TileStrategyTests(unittest.TestCase):
    def test_legacy_four_tile_matches_existing_origins(self) -> None:
        plan = LegacyFourTileStrategy().plan(
            input_shape=(4, 4, 2),
            patch_size=2,
            stride=2,
        )

        self.assertEqual(plan.strategy_id, "legacy_four_tile")
        self.assertEqual(
            [region.origin for region in plan.regions],
            [(0, 0, 0), (0, 2, 0), (2, 0, 0), (2, 2, 0)],
        )
        self.assertEqual(
            [region.slices for region in plan.regions],
            [
                (slice(0, 2), slice(0, 2), slice(0, 2)),
                (slice(0, 2), slice(2, 4), slice(0, 2)),
                (slice(2, 4), slice(0, 2), slice(0, 2)),
                (slice(2, 4), slice(2, 4), slice(0, 2)),
            ],
        )

    def test_sliding_window_covers_divisible_volume(self) -> None:
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(6, 6, 6),
            patch_size=3,
            stride=3,
        )

        self.assertEqual(len(plan.regions), 8)
        self.assertIn((3, 3, 3), [region.origin for region in plan.regions])

    def test_sliding_window_aligns_last_tile_to_boundary(self) -> None:
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(7, 7, 7),
            patch_size=3,
            stride=3,
        )

        origins = [region.origin for region in plan.regions]
        self.assertIn((4, 4, 4), origins)
        self.assertEqual(max(origin[0] for origin in origins), 4)
        self.assertEqual(max(origin[1] for origin in origins), 4)
        self.assertEqual(max(origin[2] for origin in origins), 4)

    def test_sliding_window_handles_small_volume(self) -> None:
        plan = SlidingWindowTileStrategy().plan(
            input_shape=(2, 2, 2),
            patch_size=3,
            stride=1,
        )

        self.assertEqual(len(plan.regions), 1)
        self.assertEqual(plan.regions[0].origin, (0, 0, 0))
        self.assertEqual(plan.regions[0].size, (2, 2, 2))

    def test_resolver_defaults_to_legacy_and_accepts_sliding_window(self) -> None:
        resolver = TileStrategyResolver()

        self.assertIsInstance(resolver.resolve({}), LegacyFourTileStrategy)
        self.assertIsInstance(
            resolver.resolve({"tile_strategy": "sliding_window"}),
            SlidingWindowTileStrategy,
        )


if __name__ == "__main__":
    unittest.main()
