from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..widgets.feature_range import FeatureRangeWidget
from .base import PluginPanel


def _apply_soft_combo_style(combo: QComboBox) -> None:
    combo.setObjectName("softCombo")
    if combo.view() is not None:
        combo.view().setObjectName("softComboPopup")


class XaiFamilyPluginPanel(PluginPanel):
    def __init__(
        self,
        family_id: str,
        title: str,
        description: str,
        class_label: str = "Class",
        parent=None,
    ) -> None:
        super().__init__(title, description, parent)
        self.family_id = family_id
        self._method_options_by_id: dict[str, dict[str, object]] = {}
        self._parameter_widgets: dict[str, QWidget] = {}
        self._feature_size = 0

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Data"))
        self.dataset_combo = QComboBox()
        _apply_soft_combo_style(self.dataset_combo)
        dataset_layout.addWidget(self.dataset_combo, 1)
        self.content_layout.addLayout(dataset_layout)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel(class_label))
        self.class_spinbox = QSpinBox()
        self.class_spinbox.setRange(0, 100)
        class_layout.addWidget(self.class_spinbox, 1)
        self.content_layout.addLayout(class_layout)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        _apply_soft_combo_style(self.method_combo)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self.method_combo, 1)
        self.content_layout.addLayout(method_layout)

        objective_layout = QHBoxLayout()
        objective_layout.addWidget(QLabel("Objective"))
        self.objective_combo = QComboBox()
        _apply_soft_combo_style(self.objective_combo)
        objective_layout.addWidget(self.objective_combo, 1)
        self.objective_row = QWidget()
        self.objective_row.setLayout(objective_layout)
        self.content_layout.addWidget(self.objective_row)

        self.layer_feature_group = QGroupBox("Layer / Feature")
        layer_feature_layout = QVBoxLayout(self.layer_feature_group)

        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Layer"))
        self.layer_combo = QComboBox()
        _apply_soft_combo_style(self.layer_combo)
        layer_layout.addWidget(self.layer_combo, 1)
        layer_feature_layout.addLayout(layer_layout)

        self.feature_widget = FeatureRangeWidget()
        layer_feature_layout.addWidget(self.feature_widget)
        self.layer_feature_group.setEnabled(False)
        self.content_layout.addWidget(self.layer_feature_group)

        self.params_group = QGroupBox("Parameters")
        self.params_layout = QVBoxLayout(self.params_group)
        self.params_group.setVisible(False)
        self.content_layout.addWidget(self.params_group)

        self.run_button = QPushButton("Run")
        self.content_layout.addWidget(self.run_button)
        self.content_layout.addStretch()

    def set_method_options(
        self, options: list[dict[str, object]], selected: str | None
    ) -> None:
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self._method_options_by_id = {str(option["id"]): dict(option) for option in options}
        for option in options:
            self.method_combo.addItem(str(option["name"]), str(option["id"]))
        if selected:
            index = self.method_combo.findData(selected)
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        self.method_combo.blockSignals(False)
        self._on_method_changed()

    def set_objective_options(
        self, options: list[dict[str, object]], selected: str | None
    ) -> None:
        self.objective_combo.blockSignals(True)
        self.objective_combo.clear()
        for option in options:
            self.objective_combo.addItem(str(option["name"]), str(option["id"]))
        if selected:
            index = self.objective_combo.findData(selected)
            if index >= 0:
                self.objective_combo.setCurrentIndex(index)
        self.objective_combo.blockSignals(False)

    def set_dataset_options(
        self, options: list[dict[str, str]], selected: str | None
    ) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for option in options:
            self.dataset_combo.addItem(option["name"], option["id"])
        if selected:
            index = self.dataset_combo.findData(selected)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        self.dataset_combo.blockSignals(False)

    def set_layer_options(
        self, layer_names: list[str], selected: str, feature_size: int
    ) -> None:
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItems(layer_names)
        self.layer_combo.setCurrentText(selected)
        self.layer_combo.blockSignals(False)
        self._feature_size = max(0, int(feature_size))
        self.feature_widget.set_size(feature_size)
        self._sync_capability_controls()

    def selected_dataset(self) -> str:
        return str(self.dataset_combo.currentData() or "")

    def selected_method(self) -> str:
        return str(self.method_combo.currentData() or "")

    def selected_objective(self) -> str:
        return str(self.objective_combo.currentData() or "predicted_target_mask")

    def selected_layer(self) -> str:
        return self.layer_combo.currentText()

    def selected_class(self) -> int:
        return int(self.class_spinbox.value())

    def feature_range(self) -> tuple[int, int]:
        return self.feature_widget.get_range()

    def selected_method_uses_layer_controls(self) -> bool:
        return bool(self.current_method_option().get("uses_layer_controls", True))

    def selected_method_uses_objective(self) -> bool:
        return bool(self.current_method_option().get("uses_objective", True))

    def selected_method_params(self) -> dict[str, object]:
        values = {}
        for parameter_id, widget in self._parameter_widgets.items():
            if isinstance(widget, QCheckBox):
                values[parameter_id] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[parameter_id] = widget.value()
            elif isinstance(widget, QComboBox):
                values[parameter_id] = widget.currentData()
            elif isinstance(widget, QLineEdit):
                values[parameter_id] = widget.text()
        return values

    def current_method_option(self) -> dict[str, object]:
        method_id = self.selected_method()
        return self._method_options_by_id.get(method_id, {})

    def _on_method_changed(self, *_args) -> None:
        self._rebuild_parameter_controls()
        self._sync_capability_controls()

    def _sync_capability_controls(self) -> None:
        uses_layer = self.selected_method_uses_layer_controls()
        can_select_layer = bool(uses_layer) and self.layer_combo.count() > 0
        can_adjust_feature = can_select_layer and self._feature_size > 0
        self.layer_feature_group.setVisible(bool(uses_layer))
        self.layer_feature_group.setEnabled(can_select_layer)
        self.layer_combo.setEnabled(can_select_layer)
        self.feature_widget.setEnabled(can_adjust_feature)
        self.objective_row.setVisible(self.selected_method_uses_objective())

    def _rebuild_parameter_controls(self) -> None:
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._parameter_widgets = {}
        parameters = self.current_method_option().get("parameters", [])
        if not isinstance(parameters, list):
            parameters = []
        for spec in parameters:
            if not isinstance(spec, dict):
                continue
            parameter_id = str(spec.get("id", ""))
            if not parameter_id:
                continue
            row = QHBoxLayout()
            label = QLabel(str(spec.get("label", parameter_id)))
            row.addWidget(label)
            widget = self._create_parameter_widget(spec)
            tooltip = str(spec.get("tooltip", ""))
            if tooltip:
                label.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            row.addWidget(widget, 1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            self.params_layout.addWidget(row_widget)
            self._parameter_widgets[parameter_id] = widget
        self.params_group.setVisible(bool(self._parameter_widgets))

    def _create_parameter_widget(self, spec: dict[str, object]) -> QWidget:
        kind = str(spec.get("kind", "text"))
        default = spec.get("default")
        if kind == "int":
            widget = QSpinBox()
            widget.setRange(int(spec.get("min", -999999)), int(spec.get("max", 999999)))
            widget.setSingleStep(int(spec.get("step", 1) or 1))
            widget.setValue(int(default or 0))
            return widget
        if kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(
                float(spec.get("min", -999999.0)),
                float(spec.get("max", 999999.0)),
            )
            widget.setSingleStep(float(spec.get("step", 0.1) or 0.1))
            widget.setValue(float(default or 0.0))
            return widget
        if kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
            return widget
        if kind == "choice":
            widget = QComboBox()
            for choice in spec.get("choices", []):
                if isinstance(choice, dict):
                    widget.addItem(str(choice.get("label", "")), choice.get("value"))
            index = widget.findData(default)
            if index >= 0:
                widget.setCurrentIndex(index)
            return widget
        widget = QLineEdit()
        widget.setText("" if default is None else str(default))
        return widget
