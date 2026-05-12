from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from .base import PluginPanel


class PerturbationPluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Perturbation-based XAI",
            "Run perturbation explainability on loaded data and publish a shared result volume.",
            parent,
        )

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Data"))
        self.dataset_combo = QComboBox()
        dataset_layout.addWidget(self.dataset_combo, 1)
        self.content_layout.addLayout(dataset_layout)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Answer"))
        self.class_spinbox = QSpinBox()
        self.class_spinbox.setRange(0, 100)
        class_layout.addWidget(self.class_spinbox, 1)
        self.content_layout.addLayout(class_layout)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        method_layout.addWidget(self.method_combo, 1)
        self.content_layout.addLayout(method_layout)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Block Size"))
        self.block_size_spinbox = QSpinBox()
        self.block_size_spinbox.setRange(1, 256)
        self.block_size_spinbox.setValue(16)
        size_layout.addWidget(self.block_size_spinbox, 1)
        self.content_layout.addLayout(size_layout)

        stride_layout = QHBoxLayout()
        stride_layout.addWidget(QLabel("Stride"))
        self.stride_spinbox = QSpinBox()
        self.stride_spinbox.setRange(1, 256)
        self.stride_spinbox.setValue(8)
        stride_layout.addWidget(self.stride_spinbox, 1)
        self.content_layout.addLayout(stride_layout)

        self.run_button = QPushButton("Run")
        self.content_layout.addWidget(self.run_button)
        self.content_layout.addStretch()
