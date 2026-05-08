from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QSpinBox,
)

from .base import PluginPanel


class RoiAnnotationPluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "ROI Annotation",
            "Create points and ROI boxes across 2D slices and the shared 3D workspace.",
            parent,
        )

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Off", "off")
        self.mode_combo.addItem("Point", "point")
        self.mode_combo.addItem("Box", "box")
        mode_layout.addWidget(self.mode_combo, 1)
        self.content_layout.addLayout(mode_layout)

        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("Point Size"))
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(2, 32)
        self.point_size_slider.setValue(8)
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(2, 32)
        self.point_size_spinbox.setValue(8)
        point_size_layout.addWidget(self.point_size_slider, 1)
        point_size_layout.addWidget(self.point_size_spinbox)
        self.content_layout.addLayout(point_size_layout)

        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("ROI Box"))
        self.roi_box_combo = QComboBox()
        self.roi_box_combo.addItem("None", None)
        roi_layout.addWidget(self.roi_box_combo, 1)
        self.content_layout.addLayout(roi_layout)

        action_row = QHBoxLayout()
        self.import_button = QPushButton("Import JSON")
        self.export_button = QPushButton("Export JSON")
        action_row.addWidget(self.import_button)
        action_row.addWidget(self.export_button)
        self.content_layout.addLayout(action_row)

        edit_row = QHBoxLayout()
        self.delete_selected_button = QPushButton("Delete Selected")
        self.clear_all_button = QPushButton("Clear All")
        edit_row.addWidget(self.delete_selected_button)
        edit_row.addWidget(self.clear_all_button)
        self.content_layout.addLayout(edit_row)

        self.annotation_list = QListWidget()
        self.annotation_list.setObjectName("annotationList")
        self.content_layout.addWidget(self.annotation_list, 1)
