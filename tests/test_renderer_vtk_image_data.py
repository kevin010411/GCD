import unittest

import numpy as np

from src.gcd.infrastructure.renderer import _build_vtk_image_data


class VtkImageDataConstructionTests(unittest.TestCase):
    def _build(self, data):
        return _build_vtk_image_data(
            data,
            spacing=(1.5, 2.0, 2.5),
            origin=(3.0, 4.0, 5.0),
            metadata={},
        )

    def test_supported_dtypes_are_preserved_and_share_the_numpy_buffer(self) -> None:
        from vtkmodules.util import numpy_support

        for dtype in (np.float32, np.uint8, np.uint16, np.int16):
            with self.subTest(dtype=dtype):
                source = np.arange(24, dtype=dtype).reshape(2, 3, 4)
                backing, image_data, spacing, origin = self._build(source)
                scalars = image_data.GetPointData().GetScalars()
                vtk_values = numpy_support.vtk_to_numpy(scalars)

                self.assertEqual(backing.dtype, np.dtype(dtype))
                self.assertEqual(scalars.GetDataType(), numpy_support.get_vtk_array_type(dtype))
                self.assertEqual(image_data.GetDimensions(), (4, 3, 2))
                self.assertEqual(spacing, (1.5, 2.0, 2.5))
                self.assertEqual(origin, (3.0, 4.0, 5.0))
                self.assertTrue(np.shares_memory(backing, source))
                self.assertTrue(
                    np.shares_memory(
                        image_data._numpy_backing_array, backing.reshape(-1)
                    )
                )

                source.flat[0] = 17
                self.assertEqual(vtk_values[0], 17)

    def test_float64_and_unsupported_dtypes_are_converted_to_float32(self) -> None:
        from vtkmodules.util import numpy_support

        for source in (
            np.array([[[1.25, 2.5]]], dtype=np.float64),
            np.array([[[True, False]]], dtype=np.bool_),
        ):
            with self.subTest(dtype=source.dtype):
                backing, image_data, _spacing, _origin = self._build(source)
                scalars = image_data.GetPointData().GetScalars()

                self.assertEqual(backing.dtype, np.dtype(np.float32))
                self.assertEqual(
                    scalars.GetDataType(), numpy_support.get_vtk_array_type(np.float32)
                )
                np.testing.assert_array_equal(
                    numpy_support.vtk_to_numpy(scalars), backing.reshape(-1)
                )

    def test_non_contiguous_input_is_made_contiguous_without_changing_values(self) -> None:
        source = np.arange(48, dtype=np.uint16).reshape(4, 3, 4)[::2]

        backing, image_data, _spacing, _origin = self._build(source)

        self.assertTrue(backing.flags.c_contiguous)
        self.assertEqual(image_data.GetDimensions(), (4, 3, 2))
        np.testing.assert_array_equal(backing, source)


if __name__ == "__main__":
    unittest.main()
