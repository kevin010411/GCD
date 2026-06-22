from __future__ import annotations

import unittest

import numpy as np

from src.gcd.infrastructure.renderer import VtkVolumeRenderer, _vtk_direction_matrix
from src.gcd.infrastructure.volume_loading import (
    build_display_metadata,
    shift_affine_for_padding,
)


class VolumeMetadataTests(unittest.TestCase):
    def test_build_display_metadata_preserves_origin_and_reorders_axes(self) -> None:
        affine = np.array(
            [
                [-0.36328125, 0.0, 0.0, 69.81835938],
                [0.0, -0.36328125, 0.0, 240.31835938],
                [0.0, 0.0, 0.5, -300.29998779],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        metadata = build_display_metadata(affine, (1, 2, 0))

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

        shifted = shift_affine_for_padding(affine, [3, 4, 5])

        np.testing.assert_allclose(shifted[:3, 3], np.array([7.9, 17.2, 25.0]))

    def test_vtk_direction_matrix_uses_axis_vectors_as_columns(self) -> None:
        matrix = _vtk_direction_matrix(
            {
                "vtk_direction": (
                    (-1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, -1.0, 0.0),
                )
            }
        )

        np.testing.assert_allclose(
            matrix,
            np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_render_affine_applies_vtk_direction(self) -> None:
        direction = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        affine = VtkVolumeRenderer._render_affine(
            (10.0, 20.0, 30.0), (2.0, 3.0, 4.0), direction
        )

        np.testing.assert_allclose(
            affine,
            np.array(
                [
                    [0.0, 0.0, -2.0, 10.0],
                    [-4.0, 0.0, 0.0, 20.0],
                    [0.0, 3.0, 0.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
