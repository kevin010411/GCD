from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class ModelSelectionDialog(QDialog):
    """Browse metadata without changing the active model until confirmation."""

    def __init__(self, options: list[dict[str, Any]], selected_path: str | None, parent=None):
        super().__init__(parent)
        self.setObjectName("modelSelectionDialog")
        self.setWindowTitle("Select Model")
        self.resize(760, 520)
        self.options = options
        self._preferred_path = selected_path
        self.selected_path: str | None = None
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search model name, config or tags…")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        filters = QHBoxLayout()
        self.family_filter = QComboBox()
        self.family_filter.addItem("All families", None)
        for family in sorted({option["family"] for option in options}):
            self.family_filter.addItem(family, family)
        self.class_filter = QComboBox()
        self.class_filter.setToolTip("Number of model output classes, including background")
        self.class_filter.addItem("All output classes", None)
        for classes in sorted({option["output_classes"] for option in options}):
            self.class_filter.addItem(f"{classes} classes", classes)
        filters.addWidget(QLabel("Family"))
        filters.addWidget(self.family_filter, 1)
        filters.addWidget(QLabel("Output"))
        filters.addWidget(self.class_filter, 1)
        layout.addLayout(filters)
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Model", "Family", "Output classes"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use model")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.search.textChanged.connect(self.refresh_results)
        self.family_filter.currentIndexChanged.connect(self.refresh_results)
        self.class_filter.currentIndexChanged.connect(self.refresh_results)
        self.table.itemSelectionChanged.connect(self.refresh_details)
        self.table.itemDoubleClicked.connect(lambda _item: self.accept())
        self.refresh_results()

    def current_option(self) -> dict[str, Any] | None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def refresh_results(self, *_args) -> None:
        previous = self.current_option()
        preferred = previous["path"] if previous else self._preferred_path
        terms = self.search.text().casefold().split()
        family = self.family_filter.currentData()
        classes = self.class_filter.currentData()
        matches = []
        for option in self.options:
            text = " ".join([
                option["name"], option["id"], option.get("config", ""),
                option["family"], option.get("description", ""), *option.get("tags", []),
            ]).casefold()
            if (family is None or option["family"] == family) and (
                classes is None or option["output_classes"] == classes
            ) and all(term in text for term in terms):
                matches.append(option)
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(len(matches))
        selected_row = 0
        for row, option in enumerate(matches):
            for column, value in enumerate((option["name"], option["family"], str(option["output_classes"]))):
                item = QTableWidgetItem(value)
                item.setToolTip(option["name"])
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, option)
                self.table.setItem(row, column, item)
            if option["path"] == preferred:
                selected_row = row
        if matches:
            self.table.setCurrentCell(selected_row, 0)
            self.table.selectRow(selected_row)
        self.table.blockSignals(False)
        self.count_label.setText(
            f"{len(matches)} / {len(self.options)} models" if matches
            else "No matching models. Clear the search or change the filters."
        )
        self.refresh_details()

    def refresh_details(self) -> None:
        option = self.current_option()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(option is not None)
        if option is None:
            self.details.clear()
            return
        lines = [option["name"], f"Config: {option.get('config', option['path'])}"]
        if option.get("tags"):
            lines.append("Tags: " + ", ".join(option["tags"]))
        if option.get("description"):
            lines.append(option["description"])
        self.details.setText("\n".join(lines))

    def accept(self) -> None:
        option = self.current_option()
        if option is not None:
            self.selected_path = option["path"]
            super().accept()


class ModelSelectorCombo(QComboBox):
    """Keep the existing presenter selection contract with a searchable picker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.options: list[dict[str, Any]] = []
        self.setToolTip("Choose a model with search and filters")
        self.currentIndexChanged.connect(self._update_tooltip)

    def set_options(self, options: list[dict[str, Any]]) -> None:
        previous = self.currentData()
        self.blockSignals(True)
        self.clear()
        self.options = list(options)
        for option in options:
            self.addItem(option["name"], option["path"])
        index = self.findData(previous)
        if index >= 0:
            self.setCurrentIndex(index)
        self.blockSignals(False)
        self._update_tooltip()

    def _update_tooltip(self, *_args) -> None:
        self.setToolTip(f"{self.currentText()}\nClick to search and filter models")

    def showPopup(self) -> None:
        if not self.options:
            return
        dialog = ModelSelectionDialog(self.options, self.currentData(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            index = self.findData(dialog.selected_path)
            if index >= 0:
                self.setCurrentIndex(index)
        self.hidePopup()
        dialog.deleteLater()
