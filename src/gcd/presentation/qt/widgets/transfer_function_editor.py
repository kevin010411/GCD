from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ....domain import ControlPoint, DataRange, TransferFunction


class ValueAxis(QWidget):
    def __init__(self, formatter, tick_count: int = 6, parent=None) -> None:
        super().__init__(parent)
        self._formatter = formatter
        self.vmin = 0.0
        self.vmax = 1.0
        self.tick_count = max(2, int(tick_count))
        self._horizontal_padding = 8
        self.setMinimumHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_horizontal_padding(self, padding: int) -> None:
        self._horizontal_padding = max(0, int(padding))
        self.update()

    def set_range(self, vmin: float, vmax: float) -> None:
        self.vmin = float(vmin)
        self.vmax = float(vmax) if vmin != vmax else float(vmin) + 1e-6
        self.update()

    def paintEvent(self, _event) -> None:
        width = max(1, self.width())
        height = self.height()
        left = self._horizontal_padding
        right = max(left, width - self._horizontal_padding)
        axis_width = max(1, right - left)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        axis_y = height - 14
        painter.setPen(QPen(QColor("#B0B0B0")))
        painter.drawLine(left, axis_y, right, axis_y)

        painter.setPen(QPen(QColor("#888888")))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        for index in range(self.tick_count):
            ratio = index / (self.tick_count - 1)
            value = self.vmin + ratio * (self.vmax - self.vmin)
            x = int(round(left + ratio * axis_width))
            painter.drawLine(x, axis_y, x, axis_y - 6)
            label = self._formatter(value)
            rect = painter.boundingRect(0, 0, 0, 0, 0, label)
            tx = max(0, min(x - rect.width() // 2, width - rect.width()))
            painter.drawText(tx, axis_y + 12, label)


class TransferFunctionCanvas(QWidget):
    transfer_function_changed = pyqtSignal(object, object)
    HANDLE_RADIUS = 5
    PLOT_PADDING = HANDLE_RADIUS + 3

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.transfer_function = TransferFunction.overlay_preset()
        self.data_range = DataRange(0.0, 1.0)
        self.selected_index: int | None = None

    def _plot_rect(self) -> tuple[float, float, float, float]:
        left = float(self.PLOT_PADDING)
        top = float(self.PLOT_PADDING)
        right = max(left, float(self.width() - self.PLOT_PADDING))
        bottom = max(top, float(self.height() - self.PLOT_PADDING))
        return left, top, right, bottom

    def set_transfer_function(
        self,
        transfer_function: TransferFunction,
        data_range: DataRange | None = None,
        *,
        emit_change: bool = False,
    ) -> None:
        self.transfer_function = transfer_function
        if data_range is not None:
            self.data_range = data_range
        self.selected_index = None
        self.update()
        if emit_change:
            self.transfer_function_changed.emit(self.transfer_function, self.data_range)

    def current_state(self) -> tuple[TransferFunction, DataRange]:
        return self.transfer_function, self.data_range

    def set_data_range(self, data_range: DataRange, *, emit_change: bool = True) -> None:
        self.data_range = data_range
        self.update()
        if emit_change:
            self.transfer_function_changed.emit(self.transfer_function, self.data_range)

    def _point_position(self, point: ControlPoint) -> QPointF:
        left, top, right, bottom = self._plot_rect()
        return QPointF(
            left + point.position * max(right - left, 1.0),
            top + (1.0 - point.opacity) * max(bottom - top, 1.0),
        )

    def _point_from_qpoint(self, point: QPointF, color: str) -> ControlPoint:
        left, top, right, bottom = self._plot_rect()
        width = max(right - left, 1.0)
        height = max(bottom - top, 1.0)
        position = max(0.0, min((point.x() - left) / width, 1.0))
        opacity = max(0.0, min(1.0 - ((point.y() - top) / height), 1.0))
        return ControlPoint(position, color, opacity)

    def _index_at(self, clicked_point: QPointF) -> int | None:
        for index, point in enumerate(self.transfer_function.control_points):
            if (self._point_position(point) - clicked_point).manhattanLength() < 10:
                return index
        return None

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        points = self.transfer_function.control_points
        _left, _top, _right, bottom = self._plot_rect()

        for index in range(len(points) - 1):
            p1 = self._point_position(points[index])
            p2 = self._point_position(points[index + 1])
            gradient = QLinearGradient(p1.x(), 0, p2.x(), 0)
            gradient.setColorAt(0.0, QColor(points[index].color))
            gradient.setColorAt(1.0, QColor(points[index + 1].color))
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(
                QPolygonF([p1, p2, QPointF(p2.x(), bottom), QPointF(p1.x(), bottom)])
            )

        polygon = QPolygonF([QPointF(self._point_position(points[0]).x(), bottom)])
        for point in points:
            polygon.append(self._point_position(point))
        polygon.append(QPointF(self._point_position(points[-1]).x(), bottom))
        painter.setBrush(QColor(150, 150, 150, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(polygon)

        painter.setPen(QPen(QColor("#000000"), 2))
        for index in range(len(points) - 1):
            painter.drawLine(self._point_position(points[index]), self._point_position(points[index + 1]))

        for index, point in enumerate(points):
            painter.setPen(
                QPen(QColor("#FF0000") if index == self.selected_index else QColor("#0000FF"), 2)
            )
            painter.setBrush(QColor(point.color))
            painter.drawEllipse(self._point_position(point), self.HANDLE_RADIUS, self.HANDLE_RADIUS)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        clicked_point = QPointF(event.position().x(), event.position().y())
        index = self._index_at(clicked_point)
        if index is not None:
            self.selected_index = index
            self.update()
            self.setFocus()
            return
        new_point = self._point_from_qpoint(clicked_point, "#808080")
        self.transfer_function = self.transfer_function.add_point(new_point)
        self.selected_index = self.transfer_function.control_points.index(new_point)
        self.transfer_function_changed.emit(self.transfer_function, self.data_range)
        self.update()
        self.setFocus()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        clicked_point = QPointF(event.position().x(), event.position().y())
        index = self._index_at(clicked_point)
        if index is None:
            return
        current = self.transfer_function.control_points[index]
        dialog = QColorDialog()
        dialog.setCurrentColor(QColor(current.color))
        dialog.setStyleSheet("QLineEdit { min-width: 250px; }")
        if not dialog.exec():
            return
        new_color = dialog.currentColor()
        if not new_color.isValid():
            return
        updated = replace(current, color=new_color.name().upper())
        self.transfer_function = self.transfer_function.update_point(index, updated)
        self.selected_index = index
        self.transfer_function_changed.emit(self.transfer_function, self.data_range)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() != Qt.MouseButton.LeftButton or self.selected_index is None:
            return
        points = list(self.transfer_function.control_points)
        current = points[self.selected_index]
        left, top, right, bottom = self._plot_rect()
        width = max(right - left, 1.0)
        height = max(bottom - top, 1.0)
        is_first = self.selected_index == 0
        is_last = self.selected_index == len(points) - 1
        min_position = 0.0 if is_first else points[self.selected_index - 1].position
        max_position = 1.0 if is_last else points[self.selected_index + 1].position
        if is_first:
            position = 0.0
        elif is_last:
            position = 1.0
        else:
            position = max(
                min_position, min((event.position().x() - left) / width, max_position)
            )
        opacity = max(0.0, min(1.0 - ((event.position().y() - top) / height), 1.0))
        points[self.selected_index] = replace(current, position=position, opacity=opacity)
        self.transfer_function = TransferFunction.from_iterable(points)
        self.transfer_function_changed.emit(self.transfer_function, self.data_range)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() != Qt.Key.Key_Delete or self.selected_index is None:
            return
        self.transfer_function = self.transfer_function.remove_point(self.selected_index)
        self.selected_index = None
        self.transfer_function_changed.emit(self.transfer_function, self.data_range)
        self.update()


class TransferFunctionEditor(QWidget):
    transfer_function_changed = pyqtSignal(object, object)
    load_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.canvas = TransferFunctionCanvas()
        self.axis = ValueAxis(formatter=self._format_label, tick_count=9)
        self.axis.set_horizontal_padding(self.canvas.PLOT_PADDING)
        self.axis.set_range(0.0, 1.0)
        self.canvas.transfer_function_changed.connect(self._on_transfer_function_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.axis)

        buttons = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save")
        self.load_button.clicked.connect(self.load_requested.emit)
        self.save_button.clicked.connect(self.save_requested.emit)
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

    def _format_label(self, value: float) -> str:
        absolute = abs(value)
        if absolute != 0 and (absolute >= 1e7 or absolute < 1e-4):
            return f"{value:.6e}"
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    def _on_transfer_function_changed(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> None:
        self.axis.set_range(data_range.min_value, data_range.max_value)
        self.transfer_function_changed.emit(transfer_function, data_range)

    def set_transfer_function(
        self, transfer_function: TransferFunction, data_range: DataRange
    ) -> None:
        self.axis.set_range(data_range.min_value, data_range.max_value)
        self.canvas.set_transfer_function(transfer_function, data_range)

    def get_transfer_function(self) -> tuple[TransferFunction, DataRange]:
        return self.canvas.current_state()

    def set_data_range(self, data_range: DataRange) -> None:
        self.axis.set_range(data_range.min_value, data_range.max_value)
        self.canvas.set_data_range(data_range)

    def canvas_size(self) -> tuple[float, float]:
        return float(max(self.canvas.width(), 1)), float(max(self.canvas.height(), 1))

    def choose_load_path(self) -> str:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Transfer Function", "trasfer.json", "JSON Files (*.json)"
        )
        return file_name

    def choose_save_path(self) -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Transfer Function", "trasfer.json", "JSON Files (*.json)"
        )
        return file_name
