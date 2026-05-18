import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from src.gcd.domain import ControlPoint, DataRange, TransferFunction
from src.gcd.presentation.qt.widgets.transfer_function_editor import TransferFunctionEditor


class TransferFunctionPngExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_export_color_bar_ignores_transfer_opacity_alpha(self):
        transfer_function = TransferFunction.from_iterable(
            [
                ControlPoint(0.0, "#000000", 0.0),
                ControlPoint(1.0, "#FFFFFF", 1.0),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transfer.png"
            editor = TransferFunctionEditor()
            editor.export_png(str(path), transfer_function, DataRange(0.0, 1.0))

            image = QImage(str(path))

        self.assertFalse(image.isNull())
        self.assertEqual(image.width(), 1800)
        self.assertEqual(image.height(), 360)
        self.assertEqual(image.pixelColor(93, 122).alpha(), 255)
        self.assertEqual(image.pixelColor(900, 122).alpha(), 255)


if __name__ == "__main__":
    unittest.main()
