from __future__ import annotations

import unittest

import numpy as np

from src.gcd.infrastructure.core_engine import GradCamEngine


class VolumeMetadataTests(unittest.TestCase):
    def test_build_display_metadata_preserves_origin_and_reorders_axes(self) -> None:
        engine = GradCamEngine.__new__(GradCamEngine)
        engine.PERMUTE = (1, 2, 0)

        affine = np.array(
            [
                [-0.36328125, 0.0, 0.0, 69.81835938],
                [0.0, -0.36328125, 0.0, 240.31835938],
                [0.0, 0.0, 0.5, -300.29998779],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        metadata = engine._build_display_metadata(affine)

        self.assertEqual(
            metadata["origin"],
            (69.818359375, 240.318359375, -300.29998779296875),
        )
        self.assertEqual(metadata["spacing"], (0.36328125, 0.5, 0.36328125))
        self.assertEqual(
            metadata["direction"],
            (
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
            ),
        )
        self.assertEqual(metadata["vtk_spacing"], (0.36328125, 0.5, 0.36328125))
        self.assertEqual(
            metadata["vtk_direction"],
            (
                (-1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, -1.0, 0.0),
            ),
        )

    def test_shift_affine_for_padding_moves_origin_backward(self) -> None:
        affine = np.array(
            [
                [0.7, 0.0, 0.0, 10.0],
                [0.0, 0.7, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        shifted = GradCamEngine._shift_affine_for_padding(affine, [3, 4, 5])

        np.testing.assert_allclose(shifted[:3, 3], np.array([7.9, 17.2, 25.0]))


if __name__ == "__main__":
    unittest.main()
