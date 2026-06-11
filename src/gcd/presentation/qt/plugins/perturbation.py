from __future__ import annotations

from .xai_family import XaiFamilyPluginPanel


class PerturbationPluginPanel(XaiFamilyPluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "perturbation",
            "Perturbation-based XAI",
            "Run perturbation explainability on loaded data and publish a shared result volume.",
            "Answer",
            parent,
        )
