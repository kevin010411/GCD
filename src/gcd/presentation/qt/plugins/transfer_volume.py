from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QPushButton

from ..widgets.transfer_function_editor import TransferFunctionEditor
from .base import PluginPanel


class VolumeListWidget(QListWidget):
    selection_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(str, bool)
    order_changed = pyqtSignal(list)
    name_changed = pyqtSignal(str, str)
    delete_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._visibility: dict[str, bool] = {}
        self._item_names: dict[str, str] = {}
        self._eye_open_icon = self._build_eye_icon(visible=True)
        self._eye_closed_icon = self._build_eye_icon(visible=False)
        self.setObjectName("volumeList")
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setIconSize(QSize(18, 18))
        self.currentItemChanged.connect(self._emit_selection_changed)
        self.itemChanged.connect(self._emit_item_changed)
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
        self._visibility = {}
        self._item_names = {}
        selected_item: QListWidgetItem | None = None
        for item_data in items:
            display_name = str(item_data["display_name"])
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, str(item_data["id"]))
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsDragEnabled
                | Qt.ItemFlag.ItemIsEditable
            )
            visible = bool(item_data.get("visible", True))
            volume_id = str(item_data["id"])
            self._visibility[volume_id] = visible
            self._item_names[volume_id] = display_name
            self._apply_visibility_icon(item, visible)
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

    def _emit_item_changed(self, item: QListWidgetItem) -> None:
        if self._syncing:
            return
        volume_id = str(item.data(Qt.ItemDataRole.UserRole))
        current_name = item.text()
        if self._item_names.get(volume_id) != current_name:
            self._item_names[volume_id] = current_name
            self.name_changed.emit(volume_id, current_name)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        if not self._syncing:
            self.order_changed.emit(self.ordered_volume_ids())

    def mouseReleaseEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is not None and self._is_icon_click(item, event.position().toPoint()):
            self._toggle_item_visibility(item)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Delete and self.currentItem() is not None:
            volume_id = str(self.currentItem().data(Qt.ItemDataRole.UserRole))
            self.delete_requested.emit(volume_id)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and self.currentItem() is not None:
            self._toggle_item_visibility(self.currentItem())
            event.accept()
            return
        super().keyPressEvent(event)

    def _is_icon_click(self, item: QListWidgetItem, point) -> bool:
        rect = self.visualItemRect(item)
        icon_width = self.iconSize().width() + 12
        return rect.left() <= point.x() <= rect.left() + icon_width

    def _toggle_item_visibility(self, item: QListWidgetItem) -> None:
        volume_id = str(item.data(Qt.ItemDataRole.UserRole))
        visible = not self._visibility.get(volume_id, True)
        self._visibility[volume_id] = visible
        self._apply_visibility_icon(item, visible)
        self.visibility_changed.emit(volume_id, visible)

    def _apply_visibility_icon(self, item: QListWidgetItem, visible: bool) -> None:
        item.setIcon(self._eye_open_icon if visible else self._eye_closed_icon)
        item.setToolTip("Visible" if visible else "Hidden")

    def _build_eye_icon(self, visible: bool) -> QIcon:
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = QColor("#d7dde7") if visible else QColor("#8a93a3")
        pen = QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        eye_path = QPainterPath()
        eye_path.moveTo(1.8, 9.0)
        eye_path.cubicTo(4.8, 4.2, 13.2, 4.2, 16.2, 9.0)
        eye_path.cubicTo(13.2, 13.8, 4.8, 13.8, 1.8, 9.0)
        painter.drawPath(eye_path)

        if visible:
            painter.setBrush(color)
            painter.drawEllipse(QRectF(6.8, 6.8, 4.4, 4.4))
        else:
            painter.drawLine(QPointF(3.2, 15.0), QPointF(14.8, 3.0))

        painter.end()
        return QIcon(pixmap)


class TransferVolumePluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Data",
            "Manage loaded data, prediction volumes, visibility, order, and transfer functions.",
            parent,
        )

        self.target_label = QLabel("Volumes")
        self.content_layout.addWidget(self.target_label)

        self.volume_list = VolumeListWidget(self)
        self.volume_list.setMinimumHeight(120)
        self.content_layout.addWidget(self.volume_list)

        self.delete_button = QPushButton("Delete")
        self.content_layout.addWidget(self.delete_button)
        self.delete_button.clicked.connect(self._emit_delete_selected)

        self.reorder_hint = QLabel("Reorder is enabled only in ROI mode.")
        self.content_layout.addWidget(self.reorder_hint)

        self.overlay_status = QLabel("")
        self.overlay_status.setWordWrap(True)
        self.content_layout.addWidget(self.overlay_status)

        self.transfer_editor = TransferFunctionEditor(self)
        self.transfer_editor.setMinimumHeight(320)
        self.transfer_editor.setMaximumHeight(320)
        self.content_layout.addWidget(self.transfer_editor, 0)
        self.content_layout.addStretch()

    def _emit_delete_selected(self) -> None:
        volume_id = self.volume_list.selected_volume_id()
        if volume_id:
            self.volume_list.delete_requested.emit(volume_id)
