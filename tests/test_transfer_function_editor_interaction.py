import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication

from src.gcd.presentation.qt.widgets.transfer_function_editor import (
    TransferFunctionCanvas,
)


class _MouseEvent:
    def __init__(self, x: float, y: float, *, button=Qt.MouseButton.LeftButton):
        self._position = QPointF(x, y)
        self._button = button
        self.accepted = False

    def position(self):
        return self._position

    def button(self):
        return self._button

    def buttons(self):
        return self._button

    def accept(self):
        self.accepted = True


class TransferFunctionEditorInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_drag_emits_live_changes_and_finishes_on_release(self) -> None:
        canvas = TransferFunctionCanvas()
        canvas.resize(320, 220)
        live_changes = []
        finished = []
        canvas.transfer_function_changed.connect(lambda *args: live_changes.append(args))
        canvas.transfer_function_change_finished.connect(lambda *args: finished.append(args))
        start = canvas._point_position(canvas.transfer_function.control_points[0])

        canvas.mousePressEvent(_MouseEvent(start.x(), start.y()))
        canvas.mouseMoveEvent(_MouseEvent(160, 100))

        self.assertEqual(len(live_changes), 1)
        self.assertEqual(len(finished), 0)

        canvas.mouseReleaseEvent(_MouseEvent(160, 100))

        self.assertEqual(len(finished), 1)


if __name__ == "__main__":
    unittest.main()
