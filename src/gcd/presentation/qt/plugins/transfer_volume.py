from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider

from ..widgets.transfer_function_editor import TransferFunctionEditor
from .base import PluginPanel


class TransferVolumePluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Transfer + Volume",
            "Shape volume appearance, playback and capture without leaving the shared workspace.",
            parent,
        )

        self.transfer_editor = TransferFunctionEditor(self)
        self.transfer_editor.setMinimumHeight(320)
        self.transfer_editor.setMaximumHeight(320)
        self.content_layout.addWidget(self.transfer_editor, 0)

        self.speed_label = QLabel("Rotation Speed: 0.5")
        self.content_layout.addWidget(self.speed_label)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(5)
        self.content_layout.addWidget(self.speed_slider)

        rotation_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Rotation")
        self.stop_button = QPushButton("Stop Rotation")
        rotation_buttons_layout.addWidget(self.start_button)
        rotation_buttons_layout.addWidget(self.stop_button)
        self.content_layout.addLayout(rotation_buttons_layout)

        self.content_layout.addStretch()
