from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox

from ..widgets.feature_range import FeatureRangeWidget
from .base import PluginPanel


class GradCamPluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Grad-CAM Compute",
            "Run target-class explainability and drive the shared viewer workspace.",
            parent,
        )

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Data"))
        self.dataset_combo = QComboBox()
        dataset_layout.addWidget(self.dataset_combo, 1)
        self.content_layout.addLayout(dataset_layout)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Class"))
        self.class_spinbox = QSpinBox()
        self.class_spinbox.setRange(0, 100)
        class_layout.addWidget(self.class_spinbox, 1)
        self.content_layout.addLayout(class_layout)

        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Layer"))
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo, 1)
        self.content_layout.addLayout(layer_layout)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        method_layout.addWidget(self.method_combo, 1)
        self.content_layout.addLayout(method_layout)

        self.feature_widget = FeatureRangeWidget()
        self.content_layout.addWidget(self.feature_widget)

        self.run_button = QPushButton("Run")
        self.content_layout.addWidget(self.run_button)

        self.content_layout.addStretch()
