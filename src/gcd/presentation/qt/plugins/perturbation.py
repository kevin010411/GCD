from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .xai_family import XaiFamilyPluginPanel


class OrganOcclusionControls(QWidget):
    total_requested = pyqtSignal(bool)
    interventions_changed = pyqtSignal()
    organ_visibility_changed = pyqtSignal(str, bool)
    organ_selected = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list[dict[str, object]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        self.total_button = QPushButton("Run TotalSegmentator")
        self.rerun_button = QPushButton("Re-run")
        self.rerun_button.setEnabled(False)
        actions.addWidget(self.total_button, 1)
        actions.addWidget(self.rerun_button)
        layout.addLayout(actions)
        self.status_label = QLabel("No organ masks. Run TotalSegmentator first.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.advanced_checkbox = QCheckBox("Advanced organs")
        layout.addWidget(self.advanced_checkbox)
        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(["Use", "Eye", "Organ", "Mode", "HU"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            2, self.table.horizontalHeader().ResizeMode.Stretch
        )
        self.table.setMinimumHeight(260)
        layout.addWidget(self.table)
        ordering = QHBoxLayout()
        self.move_up_button = QPushButton("Move up")
        self.move_down_button = QPushButton("Move down")
        ordering.addWidget(self.move_up_button)
        ordering.addWidget(self.move_down_button)
        layout.addLayout(ordering)
        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)
        self.total_button.clicked.connect(lambda: self.total_requested.emit(False))
        self.rerun_button.clicked.connect(lambda: self.total_requested.emit(True))
        self.advanced_checkbox.toggled.connect(self._rebuild_table)
        self.table.itemSelectionChanged.connect(self._emit_selected_organ)
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))

    def set_running(self, running: bool) -> None:
        self.total_button.setEnabled(not running)
        self.rerun_button.setEnabled(bool(self._records) and not running)
        self.table.setEnabled(not running)
        self.move_up_button.setEnabled(not running)
        self.move_down_button.setEnabled(not running)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.status_label.setText(str(text))
        self.status_label.setProperty("error", bool(error))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_organs(self, records: list[dict[str, object]]) -> None:
        self._records = [dict(record) for record in records]
        for record in self._records:
            record.setdefault("enabled", False)
            record.setdefault("visible", False)
            record.setdefault("mode", "local_mean")
            record.setdefault("fill_hu", 0.0)
        self.rerun_button.setEnabled(bool(self._records))
        self.set_status(f"{len(self._records)} non-empty organ masks are ready.")
        self._rebuild_table()

    def clear_organs(self) -> None:
        self._records = []
        self.table.setRowCount(0)
        self.rerun_button.setEnabled(False)
        self.metrics_label.clear()

    def records(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._records]

    def selected_interventions(self) -> list[dict[str, object]]:
        return [
            {"organ_id": str(item["id"]), "enabled": bool(item.get("enabled", False)),
             "visible": bool(item.get("visible", True)),
             "mode": str(item.get("mode", "local_mean")),
             "fill_hu": float(item.get("fill_hu", 0.0))}
            for item in self._records
        ]

    def set_metrics(self, metrics: dict[str, float]) -> None:
        if not metrics:
            self.metrics_label.clear()
            return
        lines = [
            "Model sensitivity to the selected organ-region information:",
            f"Stability Dice: {metrics.get('stability_dice', 0.0):.3f}",
            f"Stability IoU: {metrics.get('stability_iou', 0.0):.3f}",
            f"Target volume change: {metrics.get('target_volume_change_fraction', 0.0) * 100.0:+.1f}%",
            f"Modified image: {metrics.get('modified_fraction', 0.0) * 100.0:.2f}%",
        ]
        if "ground_truth_dice_delta" in metrics:
            lines.append(f"Ground-truth Dice change: {metrics['ground_truth_dice_delta']:+.3f}")
        self.metrics_label.setText("\n".join(lines))

    def select_organ(self, organ_id: str) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 2)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == organ_id:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return

    def _visible_records(self) -> list[dict[str, object]]:
        return list(self._records) if self.advanced_checkbox.isChecked() else [
            item for item in self._records if item.get("group") == "curated"
        ]

    def _rebuild_table(self, *_args) -> None:
        current_id = self._selected_id()
        records = self._visible_records()
        self.table.blockSignals(True)
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            organ_id = str(record["id"])
            use = QCheckBox()
            use.setChecked(bool(record.get("enabled", False)))
            use.toggled.connect(lambda checked, oid=organ_id: self._update_record(oid, "enabled", checked))
            self.table.setCellWidget(row, 0, self._centered(use))
            eye = QCheckBox()
            eye.setChecked(bool(record.get("visible", True)))
            eye.setToolTip("3D visibility only; this does not change model input.")
            eye.toggled.connect(lambda checked, oid=organ_id: self._update_visibility(oid, checked))
            self.table.setCellWidget(row, 1, self._centered(eye))
            name = QTableWidgetItem(str(record.get("display_name", organ_id)))
            name.setData(Qt.ItemDataRole.UserRole, organ_id)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, name)
            mode = QComboBox()
            mode.addItem("Local mean", "local_mean")
            mode.addItem("Gaussian blur", "gaussian_blur")
            mode.addItem("Fixed HU", "fixed_hu")
            mode.setCurrentIndex(max(0, mode.findData(record.get("mode", "local_mean"))))
            self.table.setCellWidget(row, 3, mode)
            hu = QDoubleSpinBox()
            hu.setRange(-1000.0, 1000.0)
            hu.setDecimals(1)
            hu.setValue(float(record.get("fill_hu", 0.0)))
            hu.setEnabled(mode.currentData() == "fixed_hu")
            self.table.setCellWidget(row, 4, hu)
            mode.currentIndexChanged.connect(
                lambda _index, oid=organ_id, combo=mode, spin=hu: self._mode_changed(oid, combo, spin)
            )
            hu.editingFinished.connect(
                lambda oid=organ_id, spin=hu: self._update_record(
                    oid, "fill_hu", spin.value()
                )
            )
        self.table.blockSignals(False)
        if current_id:
            self.select_organ(current_id)

    @staticmethod
    def _centered(widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(widget)
        return container

    def _record(self, organ_id: str) -> dict[str, object] | None:
        return next((item for item in self._records if item.get("id") == organ_id), None)

    def _update_record(self, organ_id: str, key: str, value: object) -> None:
        record = self._record(organ_id)
        if record is None:
            return
        record[key] = value
        self.metrics_label.setText("Settings changed; run the model to refresh results.")
        self.interventions_changed.emit()

    def _update_visibility(self, organ_id: str, visible: bool) -> None:
        record = self._record(organ_id)
        if record is None:
            return
        record["visible"] = bool(visible)
        self.organ_visibility_changed.emit(organ_id, bool(visible))

    def _mode_changed(self, organ_id: str, combo: QComboBox, hu: QDoubleSpinBox) -> None:
        mode = str(combo.currentData())
        hu.setEnabled(mode == "fixed_hu")
        self._update_record(organ_id, "mode", mode)

    def _selected_id(self) -> str:
        row = self.table.currentRow()
        item = self.table.item(row, 2) if row >= 0 else None
        return str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""

    def _emit_selected_organ(self) -> None:
        organ_id = self._selected_id()
        if organ_id:
            self.organ_selected.emit(organ_id)

    def _move_selected(self, direction: int) -> None:
        organ_id = self._selected_id()
        if not organ_id:
            return
        index = next((i for i, record in enumerate(self._records) if record.get("id") == organ_id), -1)
        target = index + int(direction)
        if index < 0 or target < 0 or target >= len(self._records):
            return
        self._records[index], self._records[target] = self._records[target], self._records[index]
        self._rebuild_table()
        self.select_organ(organ_id)
        self.interventions_changed.emit()


class PerturbationPluginPanel(XaiFamilyPluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "perturbation", "Perturbation-based XAI",
            "Run perturbation explainability on loaded data and publish a shared result volume.",
            class_label="Answer", objective_label="Score", show_answer_data=True,
            show_progress=True, show_preview_controls=True, parent=parent,
        )
        self.organ_controls = OrganOcclusionControls(self)
        self.content_layout.insertWidget(self.content_layout.indexOf(self.progress_bar), self.organ_controls)
        self.organ_controls.setVisible(False)

    def _on_method_changed(self, *_args) -> None:
        super()._on_method_changed(*_args)
        is_organ = self.selected_method() == "organ_occlusion"
        self.organ_controls.setVisible(is_organ)
        self.params_group.setVisible(bool(self._parameter_widgets) and not is_organ)
        self.preview_checkbox.setVisible(self._show_preview_controls and not is_organ)
        if is_organ:
            self.run_button.setEnabled(bool(self.organ_controls.records()))

    def set_organ_masks(self, records: list[dict[str, object]]) -> None:
        self.organ_controls.set_organs(records)
        if self.selected_method() == "organ_occlusion":
            self.run_button.setEnabled(bool(records))

    def set_organ_running(self, running: bool) -> None:
        self.organ_controls.set_running(running)
        if self.selected_method() == "organ_occlusion":
            self.run_button.setEnabled(bool(self.organ_controls.records()) and not running)
