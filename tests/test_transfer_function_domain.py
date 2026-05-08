import unittest

from src.gcd.domain import ControlPoint, DataRange, TransferFunction


class TransferFunctionDomainTests(unittest.TestCase):
    def test_control_points_are_sorted(self):
        transfer_function = TransferFunction.from_iterable(
            [
                ControlPoint(0.8, "#FFFFFF", 0.8),
                ControlPoint(0.2, "#000000", 0.2),
            ]
        )
        self.assertEqual(
            [point.position for point in transfer_function.control_points],
            [0.0, 1.0],
        )

    def test_first_and_last_control_points_are_snapped_to_endpoints(self):
        transfer_function = TransferFunction.from_iterable(
            [
                ControlPoint(0.15, "#111111", 0.1),
                ControlPoint(0.4, "#777777", 0.5),
                ControlPoint(0.85, "#FFFFFF", 0.9),
            ]
        )
        self.assertEqual(transfer_function.control_points[0].position, 0.0)
        self.assertEqual(transfer_function.control_points[-1].position, 1.0)

    def test_renderer_points_follow_data_range(self):
        transfer_function = TransferFunction.from_iterable(
            [
                ControlPoint(0.0, "#000000", 0.1),
                ControlPoint(1.0, "#FFFFFF", 0.9),
            ]
        )
        colors, opacities = transfer_function.renderer_points(DataRange(10.0, 20.0))
        self.assertEqual(colors[0][0], 10.0)
        self.assertEqual(colors[-1][0], 20.0)
        self.assertEqual(opacities[0], (10.0, 0.1))
        self.assertEqual(opacities[-1], (20.0, 0.9))

    def test_data_range_from_data_supports_percentile_and_minmax(self):
        percentile_range = DataRange.from_data([[1, 2, 3]], method="percentile")
        minmax_range = DataRange.from_data([[1, 2, 3]], method="minmax")
        self.assertLessEqual(percentile_range.min_value, percentile_range.max_value)
        self.assertEqual(minmax_range.min_value, 1.0)
        self.assertEqual(minmax_range.max_value, 3.0)

    def test_base_preset_starts_transparent_and_stays_sorted(self):
        transfer_function = TransferFunction.base_preset()
        self.assertEqual(transfer_function.control_points[0].opacity, 0.0)
        self.assertEqual(
            sorted(point.position for point in transfer_function.control_points),
            [point.position for point in transfer_function.control_points],
        )


if __name__ == "__main__":
    unittest.main()
