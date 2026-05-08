from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PluginPanel(QWidget):
    def __init__(self, title_text: str, description_text: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel(title_text)
        title.setObjectName("panelTitle")
        description = QLabel(description_text)
        description.setWordWrap(True)
        description.setObjectName("panelDescription")

        layout.addWidget(title)
        layout.addWidget(description)

        self.content_layout = layout
