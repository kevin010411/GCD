from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtWidgets import QWidget

from .camera_controls import CameraControlsPluginPanel
from .gradcam import GradCamPluginPanel
from .perturbation import PerturbationPluginPanel
from .roi_annotation import RoiAnnotationPluginPanel
from .transfer_volume import TransferVolumePluginPanel

PanelFactory = Callable[[QWidget | None], QWidget]


@dataclass(frozen=True)
class PluginDefinition:
    plugin_id: str
    title: str
    button_label: str
    panel_factory: PanelFactory
    workspace_mode: str = "standard"
    controller_factory: object | None = None


DEFAULT_PLUGIN_DEFINITIONS: tuple[PluginDefinition, ...] = (
    PluginDefinition(
        "data",
        "Data",
        "Data",
        TransferVolumePluginPanel,
        "current",
    ),
    PluginDefinition(
        "camera",
        "Camera Controls",
        "Camera",
        CameraControlsPluginPanel,
        "standard",
    ),
    PluginDefinition(
        "gradcam",
        "Gradient-Based XAI",
        "Grad-CAM",
        GradCamPluginPanel,
        "standard",
    ),
    PluginDefinition(
        "perturbation",
        "Perturbation-based XAI",
        "Perturb",
        PerturbationPluginPanel,
        "standard",
    ),
    PluginDefinition(
        "roi",
        "ROI Annotation",
        "ROI",
        RoiAnnotationPluginPanel,
        "roi",
    ),
)
