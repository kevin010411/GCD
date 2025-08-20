# -*- coding: utf-8 -*-
if __name__ == "__main__":
    print("Loading program, please wait. ", end="", flush=True)


import os, sys
import torch, torch.nn.functional as F
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QSlider,
    QFileDialog,
    QTextEdit,
    QLineEdit,
    QSizePolicy,
)
from PyQt6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QPolygonF,
    QLinearGradient,
    QFont,
    QIntValidator,
)
from PyQt6.QtCore import Qt, QPointF, QThread, pyqtSignal, QTimer
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from io import StringIO

from gcd_render import VTKRenderer
from gcd_core import gcd_core


# Custom output redirection class
class ConsoleOutput(StringIO):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def write(self, text):
        self.text_edit.insertPlainText(text)
        self.text_edit.ensureCursorVisible()
        QApplication.processEvents()

    def flush(self):
        pass


# Custom QThread subclass for processing files in the background
class FileProcessor(QThread):
    update_console = pyqtSignal(str)  # update console output
    finished = pyqtSignal()  # return processed data

    def __init__(self, file_name, core):
        super().__init__()
        self.file_name = file_name
        self.core = core

    def run(self):
        try:
            self.update_console.emit(f"info: new input selected: {self.file_name}\n")
            self.core.load_and_process_input(self.file_name)
            self.finished.emit()
        except Exception as e:
            self.update_console.emit(f"Error processing file: {str(e)}\n")


class ValueAxis(QWidget):
    def __init__(self, formatter, tick_count: int = 6, parent=None):
        super().__init__(parent)
        self._fmt = formatter
        self.vmin = 0.0
        self.vmax = 1.0
        self.tick_count = max(2, int(tick_count))
        self.setMinimumHeight(28)  # 給標籤一點高度
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_range(self, vmin: float, vmax: float):
        # 避免 0 寬範圍
        if vmin == vmax:
            vmax = vmin + 1e-6
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.update()

    def _ticks(self):
        # 產生線性刻度（含兩端）
        n = self.tick_count
        xs = np.linspace(0.0, 1.0, n)
        vs = self.vmin + xs * (self.vmax - self.vmin)
        return xs, vs

    def paintEvent(self, event):
        w = max(1, self.width())
        h = self.height()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 軸線 y 位置（靠下）
        axis_y = h - 14
        p.setPen(QPen(QColor("#b0b0b0")))
        p.drawLine(0, axis_y, w, axis_y)

        # 刻度與標籤
        xs, vs = self._ticks()
        tick_h_major = 6
        p.setPen(QPen(QColor("#888888")))
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)

        for xnorm, v in zip(xs, vs):
            x = int(round(xnorm * (w - 1)))
            # 刻度線
            p.drawLine(x, axis_y, x, axis_y - tick_h_major)

            # 文字（置中顯示）
            label = self._fmt(float(v))
            rect = p.boundingRect(0, 0, 0, 0, 0, label)
            tx = x - rect.width() // 2
            ty = axis_y + tick_h_major + 6
            # 避免超出左右邊界
            if tx < 0:
                tx = 0
            if tx + rect.width() > w:
                tx = w - rect.width()
            p.drawText(tx, ty, label)


# ---- 內層：只負責畫與互動，不含控制列 ----
class TransferFunctionEditorCanvas(QWidget):
    DEFAULT_CONTROL_POINTS = [
        (QPointF(0, 200), QColor("#606060"), 0.0),
        (QPointF(68, 194), QColor("#CBCBCB"), 0.026),
        (QPointF(175, 169), QColor("#000000"), 0.154),
        (QPointF(191, 80), QColor("#B8A0FF"), 0.602),
        (QPointF(214, 69), QColor("#0099FF"), 0.654),
        (QPointF(245, 48), QColor("#FFB700"), 0.762),
        (QPointF(300, 16), QColor("#FF0000"), 0.918),
        (QPointF(400, 0), QColor("#570000"), 1.0),
    ]
    DEFAULT_CONTROL_POINTS_2 = [
        (QPointF(0, 200), QColor("#FFFFFF"), 0.0),
        (QPointF(63, 200), QColor("#000000"), 0.0),
        (QPointF(78, 140), QColor("#0184FF"), 0.298),
        (QPointF(92, 170), QColor("#00AA00"), 0.15),
        (QPointF(128, 179), QColor("#FFFF00"), 0.106),
        (QPointF(157, 97), QColor("#FFAA00"), 0.514),
        (QPointF(203, 41), QColor("#FF0000"), 0.794),
        (QPointF(400, 0), QColor("#570000"), 1.0),
    ]

    dataRangeChanged = pyqtSignal(float, float)  # ← 新增：(vmin, vmax)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setMinimumSize(400, 200)
        self.control_points = list(self.DEFAULT_CONTROL_POINTS)
        self.selected_point = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 數值域（會由外層控制列設定）
        self.data_min = 0.0
        self.data_max = 1.0

    # ======== 對外 API：設定/取得數值域 ========
    def set_data_range(self, vmin: float, vmax: float):
        if vmin == vmax:
            vmax = vmin + 1e-6
        self.data_min, self.data_max = float(vmin), float(vmax)

        self.update()
        self.main_window.apply_transfer_functions(False)
        self.dataRangeChanged.emit(self.data_min, self.data_max)  # ← 新增：通知外層

    def fit_range_to_data(
        self,
        data: list[np.ndarray],
        method: str = "percentile",
        low_q: float = 1.0,
        high_q: float = 100.0,
    ):
        """根據資料自動設定數值域。預設用百分位避免極端值。"""
        if not data:
            return

        data = np.concatenate(
            [np.ravel(np.asarray(arr)) for arr in data if arr is not None]
        )

        if data.size == 0 or not np.isfinite(data).any():
            return

        if method == "minmax":
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
        else:
            vmin = float(np.nanpercentile(data, low_q))
            vmax = float(np.nanpercentile(data, high_q))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmin == vmax:
            vmax = vmin + 1e-6
        self.set_data_range(vmin, vmax)

    # ======== 座標 <-> 數值 的小工具 ========
    def value_to_x(self, v: float) -> float:
        t = (float(v) - self.data_min) / (self.data_max - self.data_min)
        return t * self.width()

    def x_to_value(self, x: float) -> float:
        t = float(x) / max(self.width(), 1)
        return self.data_min + t * (self.data_max - self.data_min)

    # ======== 原畫圖/互動（改成用 self.width()/height()） ========
    def reset(self, mode=1):
        if mode == 1:
            self.control_points = list(self.DEFAULT_CONTROL_POINTS)
        elif mode == 2:
            self.control_points = list(self.DEFAULT_CONTROL_POINTS_2)
        # 依目前寬度重新等比調整 x（因預設點是以 400 為寬建立）
        w = max(self.width(), 1)
        self.control_points = [
            (QPointF(p.x() / 400.0 * w, p.y()), c, a)
            for (p, c, a) in self.control_points
        ]
        self.selected_point = None
        self.main_window.apply_transfer_functions()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 漸層帶
        for i in range(len(self.control_points) - 1):
            p1, c1, _ = self.control_points[i]
            p2, c2, _ = self.control_points[i + 1]
            gradient = QLinearGradient(p1.x(), 0, p2.x(), 0)
            gradient.setColorAt(0.0, c1)
            gradient.setColorAt(1.0, c2)
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            polygon = QPolygonF(
                [p1, p2, QPointF(p2.x(), self.height()), QPointF(p1.x(), self.height())]
            )
            painter.drawPolygon(polygon)

        # Opacity 區域
        painter.setBrush(QColor(150, 150, 150, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        polygon = QPolygonF()
        polygon.append(QPointF(0, self.height()))
        for point, _, _ in self.control_points:
            polygon.append(point)
        polygon.append(QPointF(self.width(), self.height()))
        painter.drawPolygon(polygon)

        # 連線 & 控制點
        pen = QPen(QColor("#000000"), 2)
        painter.setPen(pen)
        for i in range(len(self.control_points) - 1):
            painter.drawLine(self.control_points[i][0], self.control_points[i + 1][0])

        for point, color, _ in self.control_points:
            pen = QPen(
                (
                    QColor("#ff0000")
                    if point == self.selected_point
                    else QColor("#0000ff")
                ),
                2,
            )
            painter.setPen(pen)
            painter.setBrush(color)
            painter.drawEllipse(point, 5, 5)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_point = QPointF(event.position().x(), event.position().y())
            for point, color, opacity in self.control_points:
                if (point - clicked_point).manhattanLength() < 10:
                    self.selected_point = point
                    self.update()
                    self.setFocus()
                    return
            # 新增控制點（預設色），opacity 依 y / height
            new_opacity = 1.0 - (clicked_point.y() / max(self.height(), 1))
            new_color = QColor("#808080")
            self.control_points.append((clicked_point, new_color, new_opacity))
            self.control_points = sorted(self.control_points, key=lambda p: p[0].x())
            self.selected_point = clicked_point
            self.update()
            self.setFocus()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_point = QPointF(event.position().x(), event.position().y())
            from PyQt6.QtWidgets import QColorDialog

            for i, (point, color, opacity) in enumerate(self.control_points):
                if (point - clicked_point).manhattanLength() < 10:
                    dialog = QColorDialog()
                    dialog.setCurrentColor(color)
                    dialog.setStyleSheet("QLineEdit { min-width: 250px; }")
                    dialog.exec()
                    new_color = dialog.currentColor()
                    if new_color.isValid():
                        self.control_points[i] = (point, new_color, opacity)
                        self.main_window.apply_transfer_functions(False)
                        self.update()
                    break
            # for x in self.control_points:
            #     print(
            #         f"{x[0]} {x[1].red():02X},{x[1].green():02X},{x[1].blue():02X},{x[1].alpha():02X}, {x[2]}"
            #     )

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.selected_point:
            h = max(self.height(), 1)
            w = max(self.width(), 1)
            for i, (point, color, opacity) in enumerate(self.control_points):
                if point == self.selected_point:
                    xmin = 0 if i == 0 else self.control_points[i - 1][0].x()
                    xmax = (
                        w
                        if i == len(self.control_points) - 1
                        else self.control_points[i + 1][0].x()
                    )
                    x = max(min(event.position().x(), xmax), xmin)
                    y = max(min(event.position().y(), h), 0)
                    new_opacity = 1.0 - (y / h)
                    self.control_points[i] = (QPointF(x, y), color, new_opacity)
                    self.selected_point = QPointF(x, y)
                    self.main_window.apply_transfer_functions(False)
                    self.update()
                    break

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.selected_point:
            if len(self.control_points) > 1:
                self.control_points = [
                    (point, color, opacity)
                    for point, color, opacity in self.control_points
                    if point != self.selected_point
                ]
                self.selected_point = None
                self.main_window.apply_transfer_functions()
                self.update()

    # 給外界取用：回傳 (value, color, opacity)（非像素 x）
    def export_value_points(self):
        vals = []
        for p, c, a in sorted(self.control_points, key=lambda pp: pp[0].x()):
            v = self.x_to_value(p.x())
            vals.append((v, QColor(c), float(a)))
        return vals


# ---- 外層：含底部最小/最大值控制列 + 自動帶入 ----
class TransferFunctionEditor(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.canvas = TransferFunctionEditorCanvas(main_window)

        self.minLabel = QLabel(self._fmt(self.canvas.data_min))
        self.maxLabel = QLabel(self._fmt(self.canvas.data_max))

        self.axis = ValueAxis(formatter=self._fmt, tick_count=6)
        self._on_range_changed(self.canvas.data_min, self.canvas.data_max)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.axis)

        self.canvas.dataRangeChanged.connect(self._on_range_changed)

    # 供你外部呼叫：重設預設點
    def reset(self, mode=1):
        self.canvas.reset(mode)

    # 供你外部呼叫：載入資料後自動設定範圍（預設 1~99 百分位）
    def auto_set_range_from_data(
        self,
        data: list[np.ndarray],
        method: str = "percentile",
        low_q: float = 1.0,
        high_q: float = 99.0,
    ):
        self.canvas.fit_range_to_data(data, method, low_q, high_q)
        # 讓 spinbox 反映畫布的數值域

    # 供你外部呼叫：手動設定範圍
    def set_range(self, vmin: float, vmax: float, rescale_points: bool = True):
        self.canvas.set_data_range(vmin, vmax, rescale_points)

    # 讓外界取得 (value, color, opacity) 列表
    def export_value_points(self):
        return self.canvas.export_value_points()

    def _on_range_changed(self, vmin: float, vmax: float):
        self.axis.set_range(vmin, vmax)

    def _fmt(self, v: float) -> str:
        # 友善的數字顯示：整數顯示無小數，否則最多 6 位；超大/超小用科學記號
        av = abs(v)
        if av != 0 and (av >= 1e7 or av < 1e-4):
            return f"{v:.6e}"
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}"
        return f"{v:.6f}".rstrip("0").rstrip(".")


class Feature(QWidget):
    def __init__(self, parent_layout, callback):
        super().__init__()
        self.callback = callback

        self.edit_n1 = QLineEdit()
        self.edit_n2 = QLineEdit()
        self.edit_n1.setValidator(QIntValidator(0, 999))
        self.edit_n2.setValidator(QIntValidator(0, 999))

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Feature Selection: "))

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.edit_n1, 1)
        input_layout.addWidget(QLabel(":"), 0)
        input_layout.addWidget(self.edit_n2, 1)

        self.apply_button = QPushButton("Apply")
        self.left_button = QPushButton("<")
        self.right_button = QPushButton(">")
        self.apply_button.clicked.connect(self.apply_values)
        self.left_button.clicked.connect(self.shift_left)
        self.right_button.clicked.connect(self.shift_right)
        input_layout.addWidget(self.apply_button, 1)
        input_layout.addWidget(self.left_button, 1)
        input_layout.addWidget(self.right_button, 1)

        self.apply_button.setObjectName("applyButton")
        self.left_button.setObjectName("leftButton")
        self.right_button.setObjectName("rightButton")

        layout.addLayout(input_layout)
        parent_layout.addLayout(layout)

    def apply_values(self):

        try:
            n1 = int(self.edit_n1.text()) if self.edit_n1.text() else 0
        except ValueError:
            n1 = 0
        try:
            n2 = int(self.edit_n2.text()) if self.edit_n2.text() else self.size
        except ValueError:
            n2 = self.size

        n1 = max(0, min(n1, self.size - 1))
        n2 = max(n1 + 1, min(n2, self.size))

        self.n1, self.n2 = n1, n2
        self.edit_n1.setText(f"{n1}")
        self.edit_n2.setText(f"{n2}")
        self.callback()

        print(f"Displaying feature range: [{self.n1}, {self.n2}]")

    def shift_left(self):

        try:
            n1 = int(self.edit_n1.text()) if self.edit_n1.text() else 0
        except ValueError:
            n1 = 0
        try:
            n2 = int(self.edit_n2.text()) if self.edit_n2.text() else self.size
        except ValueError:
            n2 = self.size

        n1_new, n2_new = (n1 - (n2 - n1), n1)
        if n1_new < 0:
            n1_new, n2_new = (0, n2 - n1)

        self.n1, self.n2 = n1_new, n2_new
        self.edit_n1.setText(f"{n1_new}")
        self.edit_n2.setText(f"{n2_new}")
        self.callback()

        print(f"Displaying feature range: [{self.n1}, {self.n2}]")

    def shift_right(self):

        try:
            n1 = int(self.edit_n1.text()) if self.edit_n1.text() else 0
        except ValueError:
            n1 = 0
        try:
            n2 = int(self.edit_n2.text()) if self.edit_n2.text() else self.size
        except ValueError:
            n2 = self.size

        n1_new, n2_new = (n2, n2 + (n2 - n1))
        if n2_new > self.size:
            n1_new, n2_new = (n1 - (n2 - self.size), self.size)

        self.n1, self.n2 = n1_new, n2_new
        self.edit_n1.setText(f"{n1_new}")
        self.edit_n2.setText(f"{n2_new}")
        self.callback()

        print(f"Displaying feature range: [{self.n1}, {self.n2}]")

    def setSize(self, size):
        self.size = size
        self.edit_n1.setText("0")
        self.edit_n2.setText(f"{size}")
        self.n1, self.n2 = 0, size

    def get(self):
        return (self.n1, self.n2)


class MainWindow(QMainWindow):
    def __init__(self, core):
        super().__init__()
        self.setWindowTitle("Grad-CAM Discoverer")
        self.setGeometry(QMainWindow().screen().geometry())
        self.current_input = ""

        # init variables
        self.core = core
        self.selected_layer = ""
        self.is_recording = False
        self.processor = None
        self.use_overlay = True

        # main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # components
        self.setup_vtk_renderer()
        self.setup_left_panel()
        self.setup_console()

        # add components to the main layout
        self.main_layout.addWidget(self.left_container, 1)
        self.main_layout.addWidget(self.right_widget, 3)
        self.vtk_renderer.setup_volume_data(
            [self.core.cam, self.core.volume_data],
            [self.core.img1_spacing, self.core.img1_spacing],
        )

        # init VTK and start it
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.vtk_renderer.render()
        self.vtk_renderer.start_rotation()

        # setup rotation speed and apply transfer function
        self.rotation_speed = 0.5
        self.vtk_renderer.set_rotation_speed(self.rotation_speed)
        self.vtk_renderer.store_initial_camera()
        self.apply_transfer_functions()
        self.transfer_editor.setFocus()

        # set feature size
        self.feature.setSize(list(self.core.layers.values())[0])

        # set file name
        self.file_name_label.setText(f"{core.file_name.split('/')[-1]}")

    def update_console_output(self, text):
        # safe update console in main thread
        self.console.append(text)

    # init VTK renderer components
    def setup_vtk_renderer(self):
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        self.vtk_widget = QVTKRenderWindowInteractor(self.right_widget)
        right_layout.addWidget(self.vtk_widget)
        self.vtk_renderer = VTKRenderer(self.vtk_widget, self)
        # ensure initialization completes
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    # init left panel
    def setup_left_panel(self):
        self.left_container = QWidget()
        left_container_layout = QVBoxLayout(self.left_container)

        # left widget and layout
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setSpacing(10)

        # file selector
        self.setup_file_selection()

        # transfer funciton editor
        self.transfer_editor = TransferFunctionEditor(self)
        self.left_layout.addWidget(self.transfer_editor)

        # reset button
        self.reset_buttons()

        # rotation speed control
        self.setup_rotation_controls()

        # layer selection
        self.setup_layer_selection()

        # feature selection
        self.feature = Feature(self.left_layout, self.process_layer_selection)

        # screenshot, and record buttons
        self.setup_action_buttons()

        self.left_layout.addStretch()
        left_container_layout.addWidget(self.left_widget, 3)

    # setup file selection
    def setup_file_selection(self):
        file_name_layout = QHBoxLayout()
        self.file_name_label = QLabel(f"{self.current_input.split('/')[-1]}")
        self.file_name_label.setStyleSheet(
            "font-family: Consolas; font-size: 16px; font-weight: bold;"
        )
        file_name_layout.addWidget(self.file_name_label)

        open_file_button = QPushButton("Open File")
        open_file_button.clicked.connect(self.open_file)
        file_name_layout.addWidget(open_file_button)
        self.left_layout.addLayout(file_name_layout)

    # setup rotation speed control
    def setup_rotation_controls(self):
        self.speed_label = QLabel("Rotation Speed: 0.5")
        self.left_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.update_rotation_speed)
        self.left_layout.addWidget(self.speed_slider)

        rotation_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Rotation")
        self.start_button.clicked.connect(self.vtk_renderer.start_rotation)
        self.start_button.setEnabled(False)
        rotation_buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Rotation")
        self.stop_button.clicked.connect(self.vtk_renderer.stop_rotation)
        self.stop_button.setEnabled(True)
        rotation_buttons_layout.addWidget(self.stop_button)
        self.left_layout.addLayout(rotation_buttons_layout)

    # setup layer selection
    def setup_layer_selection(self):
        self.layer_combo = QComboBox()
        layers = list(self.core.layers.keys())
        self.layer_combo.addItems(layers)
        self.layer_combo.setCurrentText(layers[0])
        self.layer_combo.currentTextChanged.connect(self.process_layer_selection)
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Layer Selection"), 1)
        layer_layout.addWidget(self.layer_combo, 2)
        self.left_layout.addLayout(layer_layout)

    # update layer selection
    def update_layer_selection(self):
        self.layer_combo.clear()
        layers = list(self.core.layers.keys())
        self.layer_combo.addItems(layers)
        self.layer_combo.setCurrentText(layers[0])

    # reset buttons
    def reset_buttons(self):
        reset_buttons_layout = QHBoxLayout()
        self.reset_button = QPushButton("Overlay")
        self.reset_button.clicked.connect(self.reset_to_overlay)
        reset_buttons_layout.addWidget(self.reset_button)

        self.reset_button2 = QPushButton("Heatmap")
        self.reset_button2.clicked.connect(self.reset_to_heatmap)
        reset_buttons_layout.addWidget(self.reset_button2)
        self.left_layout.addLayout(reset_buttons_layout)

    # setup action buttons
    def setup_action_buttons(self):
        screenshot_button = QPushButton("Save Screenshot")
        screenshot_button.clicked.connect(self.save_screenshot)
        self.left_layout.addWidget(screenshot_button)

        record_button = QPushButton("Record Video")
        record_button.clicked.connect(self.record_video)
        self.left_layout.addWidget(record_button)

        #
        # self.toggle_overlay_button = QPushButton("Switch to Heatmap Only")
        # self.toggle_overlay_button.clicked.connect(self.toggle_overlay)
        # self.left_layout.addWidget(self.toggle_overlay_button)

    # setup console
    def setup_console(self):
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        print("\r" + " " * 50 + "\r", end="", flush=True)
        sys.stdout = ConsoleOutput(self.console)
        self.left_container.layout().addWidget(self.console, 1)

    def apply_transfer_functions(self, redraw=True):
        color_points, opacity_points = self.get_transfer_functions()
        for index in range(2):
            self.vtk_renderer.set_volume_transfer_functions(
                index, color_points, opacity_points
            )
        if redraw:
            self.vtk_renderer.render()

    def get_transfer_functions(self):
        scale = 1
        color_points = [
            (px * scale, c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0)
            for px, c, _ in self.transfer_editor.canvas.export_value_points()
        ]
        opacity_points = [
            (px * scale, o)
            for px, _, o in self.transfer_editor.canvas.export_value_points()
        ]
        return color_points, opacity_points

    def process_layer_selection(self, layer=None):
        if layer:
            self.selected_layer = layer
            self.feature.setSize(self.core.layers[layer])
        else:
            layer = self.selected_layer
        self.core.compute_cam(layer, *self.feature.get(), use_overlay=self.use_overlay)
        self.transfer_editor.auto_set_range_from_data(
            [self.core.cam, self.core.volume_data], "minmax"
        )
        self.vtk_renderer.clear_volumes()
        self.vtk_renderer.setup_volume_data(
            [self.core.cam, self.core.volume_data],
            [self.core.img1_spacing, self.core.img1_spacing],
        )
        self.apply_transfer_functions()
        self.vtk_renderer.start_rotation()

    def update_rotation_speed(self, value):
        print(f"Slider value changed to: {value}", flush=True)
        self.speed_slider.valueChanged.disconnect()  # Prevent feedback loop
        try:
            # self.rotation_speed = min(value / 10.0, 5.0)  # Optional: Cap at 5.0 for smoothness
            self.rotation_speed = min(value / 10.0, 10.0)  # Cap at 10.0
            self.vtk_renderer.set_rotation_speed(self.rotation_speed)
            self.speed_label.setText(f"Rotation Speed: {self.rotation_speed:.1f}")

            if self.rotation_speed == 0.0:
                self.vtk_renderer.stop_rotation()  # This now works correctly
            else:
                self.vtk_renderer.start_rotation()  # This restarts cleanly
        finally:
            self.speed_slider.valueChanged.connect(self.update_rotation_speed)
        # print(f"Slider value after: {self.speed_slider.value()}", flush=True)

    def start_rotation(self):
        if not self.rotating and self.rotation_speed > 0:
            base_speed = 60.0
            timer_interval = 30

            def rotate_callback(obj, event):
                degrees_per_second = base_speed * self.rotation_speed
                angle_step = degrees_per_second * (timer_interval / 1000.0)
                camera = self.renderer.GetActiveCamera()
                camera.Azimuth(angle_step)
                self.renderer.ResetCameraClippingRange()
                self.render_window.Render()

            self.interactor.AddObserver("TimerEvent", rotate_callback)
            self.timer_id = self.interactor.CreateRepeatingTimer(timer_interval)
            self.rotating = True
            if self.main_window:
                self.main_window.start_button.setEnabled(False)
                self.main_window.stop_button.setEnabled(True)

    def stop_rotation(self):
        self.vtk_renderer.stop_rotation()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def reset_to_default(self, mode=1):
        self.console.append("Reset to default settings\n")
        self.transfer_editor.reset(mode)

        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.update_rotation_speed)

        self.vtk_renderer.stop_rotation()
        self.vtk_renderer.reset_camera()
        self.vtk_renderer.start_rotation()

        current_layer = self.layer_combo.currentText()
        layer_size = self.core.layers.get(current_layer, 0)
        self.feature.setSize(layer_size)

        self.process_layer_selection()

        self.transfer_editor.setFocus()

    def reset_to_overlay(self):
        self.use_overlay = True
        self.reset_to_default(1)
        # self.process_layer_selection()
        # self.reset_button.setEnabled(False)
        # self.reset_button2.setEnabled(True)

    def reset_to_heatmap(self):
        self.use_overlay = False
        self.reset_to_default(2)
        # self.process_layer_selection()
        # self.reset_button2.setEnabled(False)
        # self.reset_button.setEnabled(True)

    def save_screenshot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot", "PNG Files (*.png)"
        )
        if file_name:
            if file_name.lower().endswith(".png"):
                file_name = file_name[:-4]  # Remove .png extension if present
            self.vtk_renderer.save_screenshot(file_name)
            self.console.append(f"Screenshot saved to: {file_name}\n")

    def record_video(self):
        """Record a video of the volume rendering"""
        if not self.is_recording:
            self.is_recording = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)

            was_rotating = self.vtk_renderer.rotating
            self.vtk_renderer.stop_rotation()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)

            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Video", "rotation_video.mp4", "MP4 Files (*.mp4)"
            )
            if file_name:
                self.vtk_renderer.record_rotation_video(file_name, self.rotation_speed)

            self.is_recording = False

            if was_rotating:
                self.vtk_renderer.start_rotation()
            else:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)

    def open_file(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Open NIfTI File", "", "NIfTI Files (*.nii.gz *.nii)"
            )
            if file_name:
                print("info: clear previous VTK data", end="", flush=True)
                self.vtk_renderer.stop_rotation()
                self.vtk_renderer.clear_volumes()
                self.vtk_renderer.render()

                self.file_name_label.setText(f"{file_name.split('/')[-1]}")
                self.current_input = file_name

                self.processor = FileProcessor(file_name, self.core)
                self.processor.update_console.connect(
                    self.update_console_output, Qt.ConnectionType.QueuedConnection
                )
                self.processor.finished.connect(
                    self.on_file_processed, Qt.ConnectionType.QueuedConnection
                )
                self.processor.start()
        except Exception as e:
            print(f"Error in open_file: {str(e)}")
            self.console.append(f"Error: {str(e)}")

    def on_file_processed(self):
        # Update the main window with the processed data
        self.update_layer_selection()

        self.feature.setSize(list(self.core.layers.values())[0])
        self.core.compute_cam()
        self.transfer_editor.auto_set_range_from_data(
            [self.core.cam, self.core.volume_data]
        )
        self.vtk_renderer.setup_volume_data(
            [self.core.cam, self.core.volume_data],
            [self.core.img1_spacing, self.core.img1_spacing],
        )
        self.use_overlay = True  # Reset to overlay mode
        self.reset_to_default(1)
        # self.reset_button.setEnabled(False)
        # self.reset_button2.setEnabled(True)
        self.apply_transfer_functions()
        self.vtk_renderer.render()
        self.vtk_renderer.start_rotation()
        self.vtk_renderer.store_initial_camera()


stylesheet = """
    /* style for buttons */
    QPushButton {
        font-size: 12px;
        min-width: 120px;
        min-height: 30px;
        border-radius: 12px;
        background-color: #2196F3;
        color: white;
    }

    QPushButton:hover {
        background-color: #1976D2;
    }

    QPushButton:disabled {
        background-color: #B0BEC5;
    }

    QTextEdit {
        background-color: #2E2E2E;  /* dark gray background */
        color: #FFFFFF;             /* white text */
        font-family: Consolas;      /* Consolas fond */
        font-size: 12px;            /* font size */
    }
    
    QLineEdit {
        min-width: 100px;  
    }
    
    QPushButton#leftButton, QPushButton#rightButton {
        width: 25px;       
        min-width: 25px;
        max-width: 25px;
        min-height: 25px;
        max-height: 25px;
        font-size: 12px;
        border-radius: 10px;
        padding: 0px;
    }
    
    QPushButton#applyButton {
        width: 80px;
        min-width: 60px;
        max-width: 60px;
        min-height: 25px;
        max-height: 25px;
        font-size: 12px;
        border-radius: 10px;
        padding: 0px;
    }
"""


def main(argv, core=gcd_core(cfg_name="unet")):
    app = QApplication(argv)
    app.setStyleSheet(stylesheet)
    window = MainWindow(core)
    # window.show()
    window.showMaximized()
    window.raise_()
    window.activateWindow()
    app.exec()


if __name__ == "__main__":
    main(sys.argv)
