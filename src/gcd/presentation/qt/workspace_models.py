from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ViewerType(StrEnum):
    VOLUME_3D = "volume_3d"
    SLICE_2D = "slice_2d"


class SliceOrientation(StrEnum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


class SplitterOrientation(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class AnnotationMode(StrEnum):
    OFF = "off"
    POINT = "point"
    BOX = "box"


class WorkspaceMode(StrEnum):
    STANDARD = "standard"
    ROI = "roi"


@dataclass(slots=True)
class ViewerTileState:
    id: str
    viewer_type: ViewerType
    title: str
    is_visible: bool = True
    is_maximized: bool = False


@dataclass(slots=True)
class SliceViewState:
    id: str
    orientation: SliceOrientation
    slice_index: int = 0
    is_linked: bool = True
    overlay_modes: dict[str, bool] = field(
        default_factory=lambda: {
            "volume": True,
            "cam": True,
            "roi": False,
            "mask": False,
        }
    )


@dataclass(slots=True)
class PointAnnotation:
    id: str
    space: str
    position: tuple[float, float, float]
    size: int
    source_viewer_id: str


@dataclass(slots=True)
class Box2DAnnotation:
    id: str
    orientation: SliceOrientation
    slice_index: int
    rect: tuple[float, float, float, float]
    source_viewer_id: str


@dataclass(slots=True)
class Box3DAnnotation:
    id: str
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]
    is_roi_target: bool = False


@dataclass(slots=True)
class AnnotationState:
    mode: AnnotationMode = AnnotationMode.OFF
    point_size: int = 8
    points: list[PointAnnotation] = field(default_factory=list)
    boxes_2d: list[Box2DAnnotation] = field(default_factory=list)
    boxes_3d: list[Box3DAnnotation] = field(default_factory=list)
    selected_annotation_id: str | None = None
    active_roi_box_id: str | None = None


@dataclass(slots=True)
class SharedImagingState:
    renderable_items: list[object] = field(default_factory=list)
    camera_snapshot: dict[str, tuple[float, float, float] | float] | None = None
    slice_snapshot: dict[str, tuple[SliceOrientation, int]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class LayoutSlot:
    slot_id: str
    viewer_type: ViewerType
    default_orientation: SliceOrientation | None = None
    role: str | None = None


@dataclass(frozen=True, slots=True)
class LayoutNode:
    node_id: str
    orientation: SplitterOrientation
    sizes: tuple[int, ...]
    children: tuple["LayoutNode | LayoutSlot", ...]


@dataclass(frozen=True, slots=True)
class LayoutPreset:
    id: str
    title: str
    root_node: LayoutNode


@dataclass(slots=True)
class WorkspaceState:
    active_layout_id: str = "focus_3d"
    tiles: dict[str, ViewerTileState] = field(default_factory=dict)
    slot_assignments: dict[str, str] = field(default_factory=dict)
    splitter_sizes: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    slot_slice_states: dict[str, dict[str, SliceViewState]] = field(
        default_factory=dict
    )
    active_viewer_id: str | None = None
    volume_shape: tuple[int, int, int] = (0, 0, 0)
    crosshair_position: tuple[int, int, int] = (0, 0, 0)
    global_slice_link_mode: bool = True
    annotations: AnnotationState = field(default_factory=AnnotationState)


def default_layout_presets() -> tuple[LayoutPreset, ...]:
    return (
        LayoutPreset(
            id="focus_3d",
            title="Focus 3D",
            root_node=LayoutNode(
                node_id="focus_root",
                orientation=SplitterOrientation.HORIZONTAL,
                sizes=(68, 32),
                children=(
                    LayoutSlot("slot-3d-main", ViewerType.VOLUME_3D, role="primary_3d"),
                    LayoutNode(
                        node_id="focus_slices",
                        orientation=SplitterOrientation.VERTICAL,
                        sizes=(56, 44),
                        children=(
                            LayoutSlot(
                                "slot-slice-top",
                                ViewerType.SLICE_2D,
                                default_orientation=SliceOrientation.AXIAL,
                                role="slice_a",
                            ),
                            LayoutSlot(
                                "slot-slice-bottom",
                                ViewerType.SLICE_2D,
                                default_orientation=SliceOrientation.CORONAL,
                                role="slice_b",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        LayoutPreset(
            id="triple_slice",
            title="3D + Triple Slice",
            root_node=LayoutNode(
                node_id="triple_root",
                orientation=SplitterOrientation.HORIZONTAL,
                sizes=(58, 42),
                children=(
                    LayoutSlot("slot-3d-main", ViewerType.VOLUME_3D, role="primary_3d"),
                    LayoutNode(
                        node_id="triple_slices",
                        orientation=SplitterOrientation.VERTICAL,
                        sizes=(34, 33, 33),
                        children=(
                            LayoutSlot(
                                "slot-slice-top",
                                ViewerType.SLICE_2D,
                                default_orientation=SliceOrientation.AXIAL,
                                role="slice_a",
                            ),
                            LayoutSlot(
                                "slot-slice-middle",
                                ViewerType.SLICE_2D,
                                default_orientation=SliceOrientation.CORONAL,
                                role="slice_b",
                            ),
                            LayoutSlot(
                                "slot-slice-bottom",
                                ViewerType.SLICE_2D,
                                default_orientation=SliceOrientation.SAGITTAL,
                                role="slice_c",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        LayoutPreset(
            id="3d_only",
            title="3D Only",
            root_node=LayoutNode(
                node_id="3d_only_root",
                orientation=SplitterOrientation.HORIZONTAL,
                sizes=(100,),
                children=(
                    LayoutSlot("slot-3d-main", ViewerType.VOLUME_3D, role="primary_3d"),
                ),
            ),
        ),
        LayoutPreset(
            id="compare",
            title="Compare",
            root_node=LayoutNode(
                node_id="compare_root",
                orientation=SplitterOrientation.VERTICAL,
                sizes=(64, 36),
                children=(
                    LayoutNode(
                        node_id="compare_slices",
                        orientation=SplitterOrientation.HORIZONTAL,
                        sizes=(50, 50),
                        children=(
                            LayoutSlot(
                                "slot-slice-left",
                                ViewerType.SLICE_2D,
                                default_orientation=SliceOrientation.AXIAL,
                                role="slice_a",
                            ),
                            LayoutSlot(
                                "slot-slice-right",
                                ViewerType.SLICE_2D,
                                default_orientation=SliceOrientation.CORONAL,
                                role="slice_b",
                            ),
                        ),
                    ),
                    LayoutSlot("slot-3d-main", ViewerType.VOLUME_3D, role="primary_3d"),
                ),
            ),
        ),
    )


def clamp_slice_index(
    orientation: SliceOrientation, index: int, volume_shape: tuple[int, int, int]
) -> int:
    axis = orientation_axis(orientation)
    if axis >= len(volume_shape) or volume_shape[axis] <= 0:
        return 0
    return max(0, min(int(index), volume_shape[axis] - 1))


def orientation_axis(orientation: SliceOrientation) -> int:
    return {
        SliceOrientation.CORONAL: 2,
        SliceOrientation.AXIAL: 1,
        SliceOrientation.SAGITTAL: 0,
    }[orientation]


def collect_slots(node: LayoutNode) -> tuple[LayoutSlot, ...]:
    slots: list[LayoutSlot] = []
    for child in node.children:
        if isinstance(child, LayoutSlot):
            slots.append(child)
        else:
            slots.extend(collect_slots(child))
    return tuple(slots)
