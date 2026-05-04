import unittest

from src.gcd.application.services import TransferFunctionAppService
from src.gcd.domain import DataRange, TransferFunction


class TransferFunctionServiceTests(unittest.TestCase):
    def test_round_trip_preserves_points_and_range(self):
        service = TransferFunctionAppService()
        transfer_function = TransferFunction.overlay_preset()
        data_range = DataRange(-1.5, 9.5)

        payload = service.serialize(transfer_function, data_range, 400, 200)
        restored_transfer_function, restored_range = service.deserialize(
            payload, canvas_width=400, canvas_height=200
        )

        self.assertEqual(restored_range.min_value, -1.5)
        self.assertEqual(restored_range.max_value, 9.5)
        self.assertEqual(
            restored_transfer_function.control_points,
            transfer_function.control_points,
        )


if __name__ == "__main__":
    unittest.main()
