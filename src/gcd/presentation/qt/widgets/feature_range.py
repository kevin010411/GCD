from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FeatureRangeWidget(QWidget):
    apply_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.size = 1
        self.n1 = 0
        self.n2 = 1

        self.edit_n1 = QLineEdit()
        self.edit_n2 = QLineEdit()
        self.edit_n1.setValidator(QIntValidator(0, 999))
        self.edit_n2.setValidator(QIntValidator(0, 999))

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Feature Selection: "))

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.edit_n1, 1)
        input_layout.addWidget(QLabel(":"), 0)
        input_layout.addWidget(self.edit_n2, 1)

        self.apply_button = QPushButton("Apply")
        self.left_button = QPushButton("<")
        self.right_button = QPushButton(">")
        self.apply_button.setObjectName("applyButton")
        self.left_button.setObjectName("leftButton")
        self.right_button.setObjectName("rightButton")

        self.apply_button.clicked.connect(self.apply_values)
        self.left_button.clicked.connect(self.shift_left)
        self.right_button.clicked.connect(self.shift_right)

        input_layout.addWidget(self.apply_button, 1)
        input_layout.addWidget(self.left_button, 1)
        input_layout.addWidget(self.right_button, 1)
        layout.addLayout(input_layout)

    def set_size(self, size: int) -> None:
        self.size = max(1, int(size))
        self.n1, self.n2 = 0, self.size
        self.edit_n1.setText("0")
        self.edit_n2.setText(str(self.size))

    def get_range(self) -> tuple[int, int]:
        return self.n1, self.n2

    def apply_values(self) -> None:
        n1 = int(self.edit_n1.text() or 0)
        n2 = int(self.edit_n2.text() or self.size)
        n1 = max(0, min(n1, self.size - 1))
        n2 = max(n1 + 1, min(n2, self.size))
        self.n1, self.n2 = n1, n2
        self.edit_n1.setText(str(n1))
        self.edit_n2.setText(str(n2))
        self.apply_requested.emit()

    def shift_left(self) -> None:
        n1 = int(self.edit_n1.text() or 0)
        n2 = int(self.edit_n2.text() or self.size)
        n1_new, n2_new = (n1 - (n2 - n1), n1)
        if n1_new < 0:
            n1_new, n2_new = (0, n2 - n1)
        self.n1, self.n2 = n1_new, n2_new
        self.edit_n1.setText(str(n1_new))
        self.edit_n2.setText(str(n2_new))
        self.apply_requested.emit()

    def shift_right(self) -> None:
        n1 = int(self.edit_n1.text() or 0)
        n2 = int(self.edit_n2.text() or self.size)
        n1_new, n2_new = (n2, n2 + (n2 - n1))
        if n2_new > self.size:
            n1_new, n2_new = (n1 - (n2 - self.size), self.size)
        self.n1, self.n2 = n1_new, n2_new
        self.edit_n1.setText(str(n1_new))
        self.edit_n2.setText(str(n2_new))
        self.apply_requested.emit()
