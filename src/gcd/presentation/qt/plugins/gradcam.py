from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSpinBox

from ..widgets.feature_range import FeatureRangeWidget
from .base import PluginPanel


class GradCamPluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Grad-CAM Compute",
            "Run target-class explainability and drive the shared viewer workspace.",
            parent,
        )

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

        self.feature_widget = FeatureRangeWidget()
        self.content_layout.addWidget(self.feature_widget)

        self.content_layout.addStretch()
