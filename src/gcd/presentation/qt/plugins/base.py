from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class PluginPanel(QWidget):
    def __init__(self, title_text: str, description_text: str, parent=None) -> None:
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_content = QWidget()
        self.scroll_area.setWidget(self.scroll_content)
        outer_layout.addWidget(self.scroll_area)

        layout = QVBoxLayout(self.scroll_content)
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
