from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from .base import PluginPanel


class PlanePluginPanel(PluginPanel):
    """Controls for a shared, Slicer-Markups-Plane-like clipping plane."""

    state_changed = pyqtSignal(dict)
    reset_requested = pyqtSignal()
    import_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(
            "Plane Editor",
            "Edit a 3D plane by its center and rotation, preview it in the viewport, "
            "and optionally clip the rendered volumes.",
            parent,
        )

        self.enabled_check = QCheckBox("Enable plane")
        self.enabled_check.setChecked(False)
        self.enabled_check.setToolTip("Show the plane in the shared 3D viewport")
        self.content_layout.addWidget(self.enabled_check)

        self.center_spins = self._vector_controls("Center", -1_000_000.0, 1_000_000.0)
        self.rotation_spins = self._vector_controls("Rotated", -360.0, 360.0)
        self.normal_spins = self.rotation_spins
        for spin in self.rotation_spins:
            spin.setSingleStep(1.0)
            spin.setSuffix("°")

        self.show_rotation_axes_check = QCheckBox("Show rotation axes")
        self.show_rotation_axes_check.setChecked(True)
        self.content_layout.addWidget(self.show_rotation_axes_check)
        self.show_translation_arrows_check = QCheckBox(
            "Show six-way translation arrows"
        )
        self.show_translation_arrows_check.setChecked(True)
        self.content_layout.addWidget(self.show_translation_arrows_check)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Plane size"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.1, 1_000_000.0)
        self.size_spin.setDecimals(3)
        self.size_spin.setSingleStep(1.0)
        self.size_spin.setValue(100.0)
        size_layout.addWidget(self.size_spin, 1)
        self.content_layout.addLayout(size_layout)

        self.clipping_check = QCheckBox("Show clipping result")
        self.clipping_check.setToolTip("Apply the plane to volume ray-casting mappers")
        self.content_layout.addWidget(self.clipping_check)

        side_layout = QHBoxLayout()
        side_layout.addWidget(QLabel("Keep side"))
        self.keep_side_combo = QComboBox()
        self.keep_side_combo.addItem("Positive", "positive")
        self.keep_side_combo.addItem("Negative", "negative")
        side_layout.addWidget(self.keep_side_combo, 1)
        self.content_layout.addLayout(side_layout)

        action_row = QHBoxLayout()
        self.reset_button = QPushButton("Reset Plane")
        self.import_button = QPushButton("Import JSON")
        self.export_button = QPushButton("Export JSON")
        action_row.addWidget(self.reset_button)
        action_row.addWidget(self.import_button)
        action_row.addWidget(self.export_button)
        self.content_layout.addLayout(action_row)

        self.status_label = QLabel("Plane ready")
        self.status_label.setWordWrap(True)
        self.content_layout.addWidget(self.status_label)
        self.content_layout.addStretch()

        self.enabled_check.toggled.connect(self._emit_state)
        self.clipping_check.toggled.connect(self._emit_state)
        self.show_rotation_axes_check.toggled.connect(self._emit_state)
        self.show_translation_arrows_check.toggled.connect(self._emit_state)
        self.keep_side_combo.currentIndexChanged.connect(self._emit_state)
        self.size_spin.valueChanged.connect(self._emit_state)
        for spin in (*self.center_spins, *self.rotation_spins):
            spin.valueChanged.connect(self._emit_state)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.import_button.clicked.connect(self.import_requested.emit)
        self.export_button.clicked.connect(self.export_requested.emit)

    def _vector_controls(
        self, title: str, minimum: float, maximum: float
    ) -> tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]:
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        spins = []
        for axis in "XYZ":
            row.addWidget(QLabel(axis))
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setMinimumWidth(62)
            row.addWidget(spin, 1)
            spins.append(spin)
        self.content_layout.addLayout(row)
        return tuple(spins)  # type: ignore[return-value]

    def state(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled_check.isChecked()),
            "visible": bool(self.enabled_check.isChecked()),
            "center": [float(spin.value()) for spin in self.center_spins],
            "rotation": [float(spin.value()) for spin in self.rotation_spins],
            "show_rotation_axes": bool(self.show_rotation_axes_check.isChecked()),
            "show_translation_arrows": bool(
                self.show_translation_arrows_check.isChecked()
            ),
            "size": float(self.size_spin.value()),
            "clipping_enabled": bool(self.clipping_check.isChecked()),
            "keep_side": str(self.keep_side_combo.currentData() or "positive"),
        }

    def set_state(self, state: dict[str, object]) -> None:
        controls = [
            self.enabled_check,
            self.clipping_check,
            self.keep_side_combo,
            self.size_spin,
            self.show_rotation_axes_check,
            self.show_translation_arrows_check,
            *self.center_spins,
            *self.rotation_spins,
        ]
        for control in controls:
            control.blockSignals(True)
        self.enabled_check.setChecked(bool(state.get("enabled", state.get("visible", False))))
        center = state.get("center", (0.0, 0.0, 0.0))
        rotation = state.get("rotation", (0.0, 0.0, 0.0))
        for spin, value in zip(self.center_spins, center if isinstance(center, (list, tuple)) else ()):
            spin.setValue(float(value))
        for spin, value in zip(self.rotation_spins, rotation if isinstance(rotation, (list, tuple)) else ()):
            spin.setValue(float(value))
        legacy_show_axes = bool(state.get("show_axes", True))
        self.show_rotation_axes_check.setChecked(
            bool(state.get("show_rotation_axes", legacy_show_axes))
        )
        self.show_translation_arrows_check.setChecked(
            bool(state.get("show_translation_arrows", legacy_show_axes))
        )
        self.size_spin.setValue(float(state.get("size", 100.0)))
        self.clipping_check.setChecked(bool(state.get("clipping_enabled", False)))
        index = self.keep_side_combo.findData(str(state.get("keep_side", "positive")))
        self.keep_side_combo.setCurrentIndex(max(0, index))
        for control in controls:
            control.blockSignals(False)
        self._update_status()

    def _emit_state(self, *_args) -> None:
        self._update_status()
        self.state_changed.emit(self.state())

    def _update_status(self) -> None:
        if not self.enabled_check.isChecked():
            self.status_label.setText("Plane disabled")
        elif self.clipping_check.isChecked():
            self.status_label.setText("Plane visible; volume clipping enabled")
        else:
            self.status_label.setText("Plane visible; clipping disabled")
