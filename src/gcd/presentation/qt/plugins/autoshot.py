from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
)

from .base import PluginPanel


class AutoShotPluginPanel(PluginPanel):
    """Configure a deterministic, one-volume-at-a-time screenshot sequence."""

    import_camera_requested = pyqtSignal()
    refresh_requested = pyqtSignal()
    start_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(
            "AutoShot",
            "Import a camera view, choose the loaded data to show one at a time, "
            "and save a PNG for each item.",
            parent,
        )

        camera_row = QHBoxLayout()
        self.import_camera_button = QPushButton("Import Camera JSON")
        self.camera_status_label = QLabel("Using current camera")
        self.camera_status_label.setWordWrap(True)
        camera_row.addWidget(self.import_camera_button)
        camera_row.addWidget(self.camera_status_label, 1)
        self.content_layout.addLayout(camera_row)

        self.content_layout.addWidget(QLabel("Data to capture (checked items)"))
        self.volume_list = QListWidget()
        self.volume_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.volume_list.setMinimumHeight(150)
        self.content_layout.addWidget(self.volume_list)

        self.refresh_button = QPushButton("Refresh Loaded Data")
        self.content_layout.addWidget(self.refresh_button)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output folder"))
        self.output_directory_edit = QLineEdit()
        self.output_directory_edit.setPlaceholderText("Choose a folder")
        self.output_directory_button = QPushButton("Browse")
        output_row.addWidget(self.output_directory_edit, 1)
        output_row.addWidget(self.output_directory_button)
        self.content_layout.addLayout(output_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Filename prefix"))
        self.filename_prefix_edit = QLineEdit("autoshot")
        self.filename_prefix_edit.setToolTip(
            "Unsafe filename characters are replaced and existing PNGs get a numeric suffix."
        )
        name_row.addWidget(self.filename_prefix_edit, 1)
        self.content_layout.addLayout(name_row)

        self.overwrite_check = QCheckBox("Allow replacing existing PNG files")
        self.overwrite_check.setChecked(False)
        self.content_layout.addWidget(self.overwrite_check)

        self.start_button = QPushButton("Start AutoShot")
        self.start_button.setObjectName("primaryButton")
        self.content_layout.addWidget(self.start_button)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.content_layout.addWidget(self.status_label)
        self.content_layout.addStretch()

        self.import_camera_button.clicked.connect(self.import_camera_requested.emit)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.output_directory_button.clicked.connect(self._choose_output_directory)
        self.start_button.clicked.connect(self.start_requested.emit)

    def set_volume_items(
        self, items: list[dict[str, object]], selected_ids: set[str] | None = None
    ) -> None:
        selected_ids = selected_ids or set()
        previous = self.checked_volume_ids()
        keep_ids = selected_ids or previous
        self.volume_list.blockSignals(True)
        self.volume_list.clear()
        for payload in items:
            volume_id = str(payload.get("id", ""))
            if not volume_id:
                continue
            item = QListWidgetItem(str(payload.get("display_name", volume_id)))
            item.setData(Qt.ItemDataRole.UserRole, volume_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if volume_id in keep_ids
                else Qt.CheckState.Unchecked
            )
            self.volume_list.addItem(item)
        self.volume_list.blockSignals(False)

    def checked_volume_ids(self) -> list[str]:
        return [
            str(self.volume_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.volume_list.count())
            if self.volume_list.item(index).checkState() == Qt.CheckState.Checked
        ]

    def set_camera_status(self, text: str) -> None:
        self.camera_status_label.setText(text)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_running(self, running: bool) -> None:
        enabled = not running
        for control in (
            self.import_camera_button,
            self.refresh_button,
            self.output_directory_button,
            self.output_directory_edit,
            self.filename_prefix_edit,
            self.overwrite_check,
            self.start_button,
            self.volume_list,
        ):
            control.setEnabled(enabled)

    def output_directory(self) -> str:
        return self.output_directory_edit.text().strip()

    def filename_prefix(self) -> str:
        return self.filename_prefix_edit.text().strip() or "autoshot"

    def allow_overwrite(self) -> bool:
        return bool(self.overwrite_check.isChecked())

    def _choose_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose AutoShot output folder")
        if directory:
            self.output_directory_edit.setText(directory)
