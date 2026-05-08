from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem

from ..widgets.transfer_function_editor import TransferFunctionEditor
from .base import PluginPanel


class VolumeListWidget(QListWidget):
    selection_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(str, bool)
    order_changed = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._syncing = False
        self.setObjectName("volumeList")
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.currentItemChanged.connect(self._emit_selection_changed)
        self.itemChanged.connect(self._emit_visibility_changed)
        self.set_reorder_enabled(False)

    def set_reorder_enabled(self, enabled: bool) -> None:
        if enabled:
            self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
            self.setDefaultDropAction(Qt.DropAction.MoveAction)
        else:
            self.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)

    def set_volumes(self, items: list[dict[str, object]], selected_id: str | None) -> None:
        self._syncing = True
        self.clear()
        selected_item: QListWidgetItem | None = None
        for item_data in items:
            item = QListWidgetItem(str(item_data["display_name"]))
            item.setData(Qt.ItemDataRole.UserRole, str(item_data["id"]))
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if bool(item_data.get("visible", True))
                else Qt.CheckState.Unchecked
            )
            self.addItem(item)
            if item.data(Qt.ItemDataRole.UserRole) == selected_id:
                selected_item = item
        if selected_item is None and self.count() > 0:
            selected_item = self.item(0)
        if selected_item is not None:
            self.setCurrentItem(selected_item)
        self._syncing = False

    def ordered_volume_ids(self) -> list[str]:
        return [
            str(self.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.count())
        ]

    def selected_volume_id(self) -> str | None:
        item = self.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

    def _emit_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if self._syncing or current is None:
            return
        self.selection_changed.emit(str(current.data(Qt.ItemDataRole.UserRole)))

    def _emit_visibility_changed(self, item: QListWidgetItem) -> None:
        if self._syncing:
            return
        self.visibility_changed.emit(
            str(item.data(Qt.ItemDataRole.UserRole)),
            item.checkState() == Qt.CheckState.Checked,
        )

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        if not self._syncing:
            self.order_changed.emit(self.ordered_volume_ids())


class TransferVolumePluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Transfer + Volume",
            "Manage visible volumes, choose one to edit, and reorder ROI drawing priority.",
            parent,
        )

        self.target_label = QLabel("Volumes")
        self.content_layout.addWidget(self.target_label)

        self.volume_list = VolumeListWidget(self)
        self.volume_list.setMinimumHeight(120)
        self.content_layout.addWidget(self.volume_list)

        self.reorder_hint = QLabel("Reorder is enabled only in ROI mode.")
        self.content_layout.addWidget(self.reorder_hint)

        self.transfer_editor = TransferFunctionEditor(self)
        self.transfer_editor.setMinimumHeight(320)
        self.transfer_editor.setMaximumHeight(320)
        self.content_layout.addWidget(self.transfer_editor, 0)
        self.content_layout.addStretch()
