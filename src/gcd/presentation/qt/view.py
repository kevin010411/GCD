from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from ...infrastructure.renderer import VtkVolumeRenderer
from .widgets.feature_range import FeatureRangeWidget
from .widgets.transfer_function_editor import TransferFunctionEditor


class MainWindowView(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Grad-CAM Discoverer")
        self.setGeometry(QMainWindow().screen().geometry())

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_container = QWidget()
        left_container_layout = QVBoxLayout(self.left_container)
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setSpacing(10)

        self.model_combo = QComboBox(self)
        self.left_layout.addWidget(self.model_combo)

        file_name_layout = QHBoxLayout()
        self.file_name_label = QLabel("")
        self.file_name_label.setStyleSheet(
            "font-family: Consolas; font-size: 16px; font-weight: bold;"
        )
        self.open_file_button = QPushButton("Open File")
        file_name_layout.addWidget(self.file_name_label)
        file_name_layout.addWidget(self.open_file_button)
        self.left_layout.addLayout(file_name_layout)

        self.transfer_editor = TransferFunctionEditor(self)
        self.left_layout.addWidget(self.transfer_editor)

        reset_buttons_layout = QHBoxLayout()
        self.overlay_button = QPushButton("Overlay")
        self.heatmap_button = QPushButton("Heatmap")
        reset_buttons_layout.addWidget(self.overlay_button)
        reset_buttons_layout.addWidget(self.heatmap_button)
        self.left_layout.addLayout(reset_buttons_layout)

        replace_buttons_layout = QHBoxLayout()
        self.replace_camera_button = QPushButton("Replace Camera")
        replace_buttons_layout.addWidget(self.replace_camera_button)
        self.left_layout.addLayout(replace_buttons_layout)

        self.speed_label = QLabel("Rotation Speed: 0.5")
        self.left_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(5)
        self.left_layout.addWidget(self.speed_slider)

        rotation_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Rotation")
        self.stop_button = QPushButton("Stop Rotation")
        rotation_buttons_layout.addWidget(self.start_button)
        rotation_buttons_layout.addWidget(self.stop_button)
        self.left_layout.addLayout(rotation_buttons_layout)

        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Layer Selection"), 1)
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo, 2)
        self.left_layout.addLayout(layer_layout)

        self.feature_widget = FeatureRangeWidget()
        self.left_layout.addWidget(self.feature_widget)

        self.save_screenshot_button = QPushButton("Save Screenshot")
        self.record_video_button = QPushButton("Record Video")
        self.left_layout.addWidget(self.save_screenshot_button)
        self.left_layout.addWidget(self.record_video_button)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Class Selection: "))
        self.class_spinbox = QSpinBox()
        self.class_spinbox.setRange(0, 100)
        class_layout.addWidget(self.class_spinbox, 1)
        self.left_layout.addLayout(class_layout)

        self.left_layout.addStretch()
        left_container_layout.addWidget(self.left_widget, 1)

        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        self.vtk_widget = QVTKRenderWindowInteractor(self.right_widget)
        right_layout.addWidget(self.vtk_widget)

        self.main_layout.addWidget(self.left_container, 1)
        self.main_layout.addWidget(self.right_widget, 3)

        self.renderer = VtkVolumeRenderer(self.vtk_widget)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def set_model_options(self, options: list[dict[str, str]]) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for option in options:
            self.model_combo.addItem(option["name"], option["path"])
        self.model_combo.blockSignals(False)

    def set_layer_options(self, layer_names: list[str], selected: str) -> None:
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItems(layer_names)
        self.layer_combo.setCurrentText(selected)
        self.layer_combo.blockSignals(False)

    def set_feature_size(self, size: int) -> None:
        self.feature_widget.set_size(size)

    def set_file_name(self, file_name: str) -> None:
        self.file_name_label.setText(os.path.basename(file_name))

    def set_rotation_speed_label(self, speed: float) -> None:
        self.speed_label.setText(f"Rotation Speed: {speed:.1f}")

    def set_rotation_running(self, is_running: bool) -> None:
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)

    def selected_model_path(self) -> str | None:
        return self.model_combo.currentData()

    def selected_layer(self) -> str:
        return self.layer_combo.currentText()

    def selected_class(self) -> int:
        return self.class_spinbox.value()

    def feature_range(self) -> tuple[int, int]:
        return self.feature_widget.get_range()

    def choose_input_file(self) -> str:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open NIfTI File", "", "NIfTI Files (*.nii.gz *.nii)"
        )
        return file_name

    def choose_screenshot_file(self) -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot", "PNG Files (*.png)"
        )
        return file_name[:-4] if file_name.lower().endswith(".png") else file_name

    def choose_video_file(self) -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "rotation_video.mp4", "MP4 Files (*.mp4)"
        )
        return file_name
