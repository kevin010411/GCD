from __future__ import annotations

from .xai_family import XaiFamilyPluginPanel


class GradCamPluginPanel(XaiFamilyPluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "gradient",
            "Grad-CAM Compute",
            "Run target-class explainability and drive the shared viewer workspace.",
            class_label="Class",
            objective_label="Aggregation",
            parent=parent,
        )
