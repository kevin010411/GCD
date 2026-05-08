from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider

from .base import PluginPanel


class CameraControlsPluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Camera Controls",
            "Tune camera motion and quickly reframe the active 3D view.",
            parent,
        )

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

        self.reset_camera_button = QPushButton("Reset Camera")
        self.content_layout.addWidget(self.reset_camera_button)

        camera_io_layout = QHBoxLayout()
        self.import_camera_button = QPushButton("Import Camera")
        self.export_camera_button = QPushButton("Export Camera")
        camera_io_layout.addWidget(self.import_camera_button)
        camera_io_layout.addWidget(self.export_camera_button)
        self.content_layout.addLayout(camera_io_layout)
        self.content_layout.addStretch()
