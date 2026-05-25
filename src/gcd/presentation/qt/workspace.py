from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
from PyQt6.QtCore import QMimeData, QPoint, QPointF, QRectF, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDrag, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from ...domain import DataRange, TransferFunction
from ...infrastructure.renderer import StandardMultiVolumeRenderer, VtkVolumeRenderer
from .annotation_geometry import move_rect, normalize_rect, resize_rect_with_handle
from .workspace_models import (
    AnnotationMode,
    Box2DAnnotation,
    Box3DAnnotation,
    PointAnnotation,
    LayoutNode,
    LayoutPreset,
    LayoutSlot,
    SliceOrientation,
    SliceViewState,
    SplitterOrientation,
    SharedImagingState,
    ViewerTileState,
    ViewerType,
    WorkspaceMode,
    WorkspaceState,
    clamp_slice_index,
    collect_slots,
    default_layout_presets,
    orientation_axis,
)


def _as_numpy(volume) -> np.ndarray | None:
    if volume is None:
        return None
    if hasattr(volume, "detach"):
        return np.array(volume.detach().cpu().numpy(), copy=False)
    return np.array(volume, copy=False)


def _color_map_from_transfer_function(
    transfer_function: TransferFunction, data_range: DataRange
) -> np.ndarray:
    colors, _ = transfer_function.renderer_points(data_range)
    if not colors:
        return np.tile(np.linspace(0, 255, 256, dtype=np.uint8)[:, None], (1, 3))

    values = np.array([item[0] for item in colors], dtype=np.float32)
    rgb = np.array([[item[1], item[2], item[3]] for item in colors], dtype=np.float32)
    sample_points = np.linspace(values[0], values[-1], 256, dtype=np.float32)
    mapped = np.empty((256, 3), dtype=np.float32)
    for index in range(3):
        mapped[:, index] = np.interp(sample_points, values, rgb[:, index])
    return np.clip(mapped * 255.0, 0, 255).astype(np.uint8)


def _extract_slice(
    volume: np.ndarray, orientation: SliceOrientation, index: int
) -> np.ndarray:
    if orientation == SliceOrientation.AXIAL:
        return volume[:, index, :]
    if orientation == SliceOrientation.CORONAL:
        return volume[:, :, index]
    return volume[index, :, :]


def _normalize_slice(slice_array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(
        slice_array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    min_value = float(array.min()) if array.size else 0.0
    max_value = float(array.max()) if array.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _blend_slice_image(
    volume_slice: np.ndarray | None,
    xai_slices: list[tuple[np.ndarray, np.ndarray]],
    slice_state: SliceViewState,
) -> np.ndarray:
    if volume_slice is None and not xai_slices:
        return np.zeros((32, 32, 3), dtype=np.uint8)

    base = None
    if slice_state.overlay_modes.get("volume", True) and volume_slice is not None:
        gray = _normalize_slice(volume_slice)
        base = np.stack([gray, gray, gray], axis=-1)

    if slice_state.overlay_modes.get("cam", True):
        for xai_slice, color_map in xai_slices:
            cam_normalized = _normalize_slice(xai_slice)
            cam_rgb = color_map[cam_normalized]
            if base is None:
                base = cam_rgb
            else:
                alpha = (cam_normalized.astype(np.float32) / 255.0)[..., None] * 0.7
                base = np.clip(base * (1.0 - alpha) + cam_rgb * alpha, 0, 255).astype(
                    np.uint8
                )

    if base is None:
        fallback = _normalize_slice(
            volume_slice if volume_slice is not None else xai_slices[0][0]
        )
        base = np.stack([fallback, fallback, fallback], axis=-1)
    if slice_state.orientation == SliceOrientation.AXIAL:
        return np.ascontiguousarray(np.flipud(np.fliplr(base)))
    if slice_state.orientation == SliceOrientation.CORONAL:
        return np.ascontiguousarray(np.flipud(np.fliplr(np.transpose(base, (1, 0, 2)))))
    return np.ascontiguousarray(np.flipud(np.fliplr(base)))


def _metadata_affine(
    item: dict[str, object], *, source: bool = False
) -> np.ndarray | None:
    key = "source_affine" if source else "metadata"
    if source:
        affine = item.get(key)
    else:
        metadata = item.get(key)
        affine = metadata.get("affine") if isinstance(metadata, dict) else None
    if affine is None:
        return None
    return np.array(affine, dtype=np.float32, copy=True)


def _slice_output_shape(
    volume_shape: tuple[int, int, int], orientation: SliceOrientation
) -> tuple[int, int]:
    if orientation == SliceOrientation.AXIAL:
        return (volume_shape[0], volume_shape[2])
    if orientation == SliceOrientation.CORONAL:
        return (volume_shape[0], volume_shape[1])
    return (volume_shape[1], volume_shape[2])


def _world_grid_for_slice(
    base_shape: tuple[int, int, int],
    base_affine: np.ndarray,
    orientation: SliceOrientation,
    index: int,
) -> np.ndarray:
    height, width = _slice_output_shape(base_shape, orientation)
    row_coords, col_coords = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    if orientation == SliceOrientation.AXIAL:
        voxel_coords = np.stack(
            [row_coords, np.full_like(row_coords, float(index)), col_coords], axis=-1
        )
    elif orientation == SliceOrientation.CORONAL:
        voxel_coords = np.stack(
            [row_coords, col_coords, np.full_like(row_coords, float(index))], axis=-1
        )
    else:
        voxel_coords = np.stack(
            [np.full_like(row_coords, float(index)), row_coords, col_coords], axis=-1
        )
    homogeneous = np.concatenate(
        [voxel_coords, np.ones((*voxel_coords.shape[:2], 1), dtype=np.float32)], axis=-1
    )
    return homogeneous @ base_affine.T


def _sample_slice_with_affine(
    volume: np.ndarray,
    world_grid: np.ndarray,
    inverse_affine: np.ndarray,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    homogeneous = world_grid.reshape(-1, 4)
    sample_voxels = homogeneous @ inverse_affine.T
    sample_voxels = sample_voxels[:, :3].reshape(*world_grid.shape[:2], 3)

    depth, height, width = volume.shape
    x = sample_voxels[..., 2]
    y = sample_voxels[..., 1]
    z = sample_voxels[..., 0]

    def _norm(values: np.ndarray, size: int) -> np.ndarray:
        if size <= 1:
            return np.zeros_like(values, dtype=np.float32)
        return ((values / float(size - 1)) * 2.0 - 1.0).astype(np.float32)

    grid = np.stack([_norm(x, width), _norm(y, height), _norm(z, depth)], axis=-1)
    grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(1)
    volume_tensor = (
        torch.from_numpy(volume.astype(np.float32, copy=False))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sampled = F.grid_sample(
        volume_tensor,
        grid_tensor,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, 0, 0].detach().cpu().numpy()


def _resample_item_slice_to_base(
    base_item: dict[str, object],
    overlay_item: dict[str, object],
    orientation: SliceOrientation,
    index: int,
) -> tuple[np.ndarray | None, str | None]:
    overlay_data = overlay_item.get("data")
    base_data = base_item.get("data")
    if overlay_data is None or base_data is None:
        return None, None
    overlay_array = np.array(overlay_data, copy=False)
    base_shape = tuple(int(v) for v in np.array(base_data, copy=False).shape)
    if tuple(int(v) for v in overlay_array.shape) == base_shape:
        base_affine = _metadata_affine(base_item)
        overlay_affine = _metadata_affine(overlay_item)
        if (
            base_affine is None
            or overlay_affine is None
            or np.allclose(base_affine, overlay_affine, atol=1e-4)
        ):
            return _extract_slice(overlay_array, orientation, index), None

    base_affine = _metadata_affine(base_item)
    overlay_affine = _metadata_affine(overlay_item)
    if base_affine is None or overlay_affine is None:
        return (
            None,
            f"{overlay_item.get('display_name', 'Heatmap')}: missing geometry metadata",
        )
    try:
        inverse_affine = np.linalg.inv(overlay_affine)
    except np.linalg.LinAlgError:
        return (
            None,
            f"{overlay_item.get('display_name', 'Heatmap')}: invalid geometry transform",
        )
    try:
        world_grid = _world_grid_for_slice(base_shape, base_affine, orientation, index)
        return (
            _sample_slice_with_affine(overlay_array, world_grid, inverse_affine),
            f"{overlay_item.get('display_name', 'Heatmap')}: resampled to active data",
        )
    except Exception:
        return None, f"{overlay_item.get('display_name', 'Heatmap')}: resample failed"


def _to_qt_orientation(orientation: SplitterOrientation) -> Qt.Orientation:
    return (
        Qt.Orientation.Horizontal
        if orientation == SplitterOrientation.HORIZONTAL
        else Qt.Orientation.Vertical
    )


@dataclass(slots=True)
class ViewerPayload:
    renderable_items: list[dict[str, object]] = field(default_factory=list)


class TileHeader(QFrame):
    drag_started = pyqtSignal(str)

    def __init__(self, viewer_id: str, title: str, parent=None) -> None:
        super().__init__(parent)
        self.viewer_id = viewer_id
        self.setObjectName("tileHeader")
        self.setFixedHeight(38)
        self._drag_start: QPoint | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("tileTitle")
        self.title_label.setMinimumHeight(18)
        layout.addWidget(self.title_label)
        layout.addStretch()

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            self._drag_start is None
            or not event.buttons() & Qt.MouseButton.LeftButton
            or (event.position().toPoint() - self._drag_start).manhattanLength() < 8
        ):
            super().mouseMoveEvent(event)
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(self.viewer_id)
        drag.setMimeData(mime)
        self.drag_started.emit(self.viewer_id)
        drag.exec(Qt.DropAction.MoveAction)
        self._drag_start = None
        super().mouseMoveEvent(event)


class ViewerTileWidget(QFrame):
    selected = pyqtSignal(str)

    def __init__(self, viewer_id: str, title: str, parent=None) -> None:
        super().__init__(parent)
        self.viewer_id = viewer_id
        self.setObjectName("viewerTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = TileHeader(viewer_id, title)
        self.header.drag_started.connect(self.selected.emit)
        layout.addWidget(self.header)
        self.body = QFrame()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 10, 10, 10)
        self.body_layout.setSpacing(8)
        layout.addWidget(self.body, 1)

    def set_content(self, widget: QWidget) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
        self.body_layout.addWidget(widget, 1)

    def set_title(self, title: str) -> None:
        self.header.set_title(title)

    def mousePressEvent(self, event) -> None:
        self.selected.emit(self.viewer_id)
        super().mousePressEvent(event)


class SlotHost(QFrame):
    swap_requested = pyqtSignal(str, str)

    def __init__(self, slot: LayoutSlot, parent=None) -> None:
        super().__init__(parent)
        self.slot = slot
        self.viewer_id: str | None = None
        self.setAcceptDrops(True)
        self.setObjectName("slotHost")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout

    def bind_tile(self, viewer_id: str, tile_widget: ViewerTileWidget) -> None:
        self.viewer_id = viewer_id
        while self._layout.count():
            item = self._layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
        self._layout.addWidget(tile_widget, 1)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasText() and event.mimeData().text() != self.viewer_id:
            self.setProperty("dropTarget", True)
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dropTarget", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        source_viewer_id = event.mimeData().text()
        if self.viewer_id and source_viewer_id and source_viewer_id != self.viewer_id:
            self.swap_requested.emit(source_viewer_id, self.viewer_id)
            event.acceptProposedAction()
        else:
            event.ignore()
        self.setProperty("dropTarget", False)
        self.style().unpolish(self)
        self.style().polish(self)


class SliceCanvasWidget(QWidget):
    point_created = pyqtSignal(str, float, float)
    box_created = pyqtSignal(str, tuple)
    box_updated = pyqtSignal(str, tuple)
    annotation_selected = pyqtSignal(str)

    def __init__(self, viewer_id: str, parent=None) -> None:
        super().__init__(parent)
        self.viewer_id = viewer_id
        self._pixmap: QPixmap | None = None
        self._image_size = (1, 1)
        self._orientation = SliceOrientation.AXIAL
        self._mode = AnnotationMode.OFF
        self._selected_annotation_id: str | None = None
        self._points: list[tuple[str, float, float, bool]] = []
        self._boxes: list[tuple[str, float, float, float, float, bool]] = []
        self._roi_projection: tuple[float, float, float, float] | None = None
        self._point_size = 8
        self._drag_start: tuple[float, float] | None = None
        self._dragging_box_id: str | None = None
        self._dragging_corner_index: int | None = None
        self._dragging_box_rect: tuple[float, float, float, float] | None = None
        self._dragging_box_offset: tuple[float, float] | None = None
        self._preview_rect: tuple[float, float, float, float] | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(180, 180)
        self.setObjectName("sliceCanvas")

    def set_render_state(
        self,
        *,
        pixmap: QPixmap | None,
        image_size: tuple[int, int],
        orientation: SliceOrientation,
        mode: AnnotationMode,
        point_size: int,
        selected_annotation_id: str | None,
        points: list[tuple[str, float, float, bool]],
        boxes: list[tuple[str, float, float, float, float, bool]],
        roi_projection: tuple[float, float, float, float] | None,
    ) -> None:
        self._pixmap = pixmap
        self._image_size = image_size
        self._orientation = orientation
        self._mode = mode
        self._point_size = point_size
        self._selected_annotation_id = selected_annotation_id
        self._points = points
        self._boxes = boxes
        self._roi_projection = roi_projection
        self.update()

    def _display_rect(self) -> QRectF:
        if self._pixmap is None:
            return QRectF(0, 0, float(self.width()), float(self.height()))
        pixmap_size = self._pixmap.size()
        scaled = pixmap_size.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        x = (self.width() - scaled.width()) / 2.0
        y = (self.height() - scaled.height()) / 2.0
        return QRectF(x, y, float(scaled.width()), float(scaled.height()))

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        rect = self._display_rect()
        width, height = self._image_size
        px = rect.left() + (x / max(1.0, width - 1)) * rect.width()
        py = rect.top() + (y / max(1.0, height - 1)) * rect.height()
        return QPointF(px, py)

    def _widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        rect = self._display_rect()
        if not rect.contains(point):
            return None
        width, height = self._image_size
        x = ((point.x() - rect.left()) / max(1.0, rect.width())) * max(1.0, width - 1)
        y = ((point.y() - rect.top()) / max(1.0, rect.height())) * max(1.0, height - 1)
        return (max(0.0, min(x, width - 1)), max(0.0, min(y, height - 1)))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090E14"))
        rect = self._display_rect()
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                rect.size().toSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(rect.topLeft(), scaled)
        else:
            painter.setPen(QColor("#8FA6C5"))
            painter.drawText(
                self.rect(),
                int(Qt.AlignmentFlag.AlignCenter),
                "Load a volume to inspect slices",
            )

        if self._roi_projection is not None:
            painter.setPen(QPen(QColor("#F0C94A"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._rect_to_widget(self._roi_projection))

        for annotation_id, x, y, selected in self._points:
            center = self._image_to_widget(x, y)
            radius = max(3.0, self._point_size / 2.0)
            painter.setPen(QPen(QColor("#F4F8FF"), 1.5))
            painter.setBrush(QBrush(QColor("#FF754A" if selected else "#4CA8FF")))
            painter.drawEllipse(center, radius, radius)

        for annotation_id, x1, y1, x2, y2, selected in self._boxes:
            color = QColor("#4CA8FF" if selected else "#45D17A")
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            widget_rect = self._rect_to_widget((x1, y1, x2, y2))
            painter.drawRect(widget_rect)
            if selected:
                for point in self._rect_handles((x1, y1, x2, y2)):
                    handle_center = self._image_to_widget(point[0], point[1])
                    painter.setBrush(QBrush(QColor("#F5F0A2")))
                    painter.drawRect(
                        QRectF(
                            handle_center.x() - 4,
                            handle_center.y() - 4,
                            8,
                            8,
                        )
                    )

        if self._preview_rect is not None:
            painter.setPen(QPen(QColor("#C7E6FF"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._rect_to_widget(self._preview_rect))

        self._draw_orientation_markers(painter, rect)

    def _draw_orientation_markers(self, painter: QPainter, rect: QRectF) -> None:
        if rect.width() <= 24 or rect.height() <= 24:
            return
        top, bottom, left, right = self._orientation_labels()
        painter.setPen(QColor("#D7E7FF"))
        font = painter.font()
        font.setPointSize(max(8, font.pointSize()))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QRectF(rect.left(), rect.top() + 4, rect.width(), 18),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            top,
        )
        painter.drawText(
            QRectF(rect.left(), rect.bottom() - 22, rect.width(), 18),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom),
            bottom,
        )
        painter.drawText(
            QRectF(rect.left() + 6, rect.top(), 18, rect.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            left,
        )
        painter.drawText(
            QRectF(rect.right() - 24, rect.top(), 18, rect.height()),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            right,
        )

    def _orientation_labels(self) -> tuple[str, str, str, str]:
        if self._orientation == SliceOrientation.AXIAL:
            return ("A", "P", "R", "L")
        if self._orientation == SliceOrientation.CORONAL:
            return ("S", "I", "A", "P")
        return ("S", "I", "R", "L")

    def _rect_to_widget(self, rect: tuple[float, float, float, float]) -> QRectF:
        x1, y1, x2, y2 = rect
        p1 = self._image_to_widget(min(x1, x2), min(y1, y2))
        p2 = self._image_to_widget(max(x1, x2), max(y1, y2))
        return QRectF(p1, p2)

    def _rect_handles(
        self, rect: tuple[float, float, float, float]
    ) -> list[tuple[float, float]]:
        x1, y1, x2, y2 = normalize_rect(rect)
        return [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
        ]

    def _find_box_hit(
        self, point: tuple[float, float]
    ) -> tuple[str | None, int | None]:
        for annotation_id, x1, y1, x2, y2, _selected in self._boxes:
            left, top, right, bottom = normalize_rect((x1, y1, x2, y2))
            handles = self._rect_handles((left, top, right, bottom))
            for index, handle in enumerate(handles):
                if abs(handle[0] - point[0]) <= 4 and abs(handle[1] - point[1]) <= 4:
                    return annotation_id, index
            if left <= point[0] <= right and top <= point[1] <= bottom:
                return annotation_id, None
        return None, None

    def _box_rect_by_id(
        self, annotation_id: str
    ) -> tuple[float, float, float, float] | None:
        for box_id, x1, y1, x2, y2, _selected in self._boxes:
            if box_id == annotation_id:
                return normalize_rect((x1, y1, x2, y2))
        return None

    def mousePressEvent(self, event) -> None:
        image_pos = self._widget_to_image(event.position())
        if image_pos is None:
            return
        annotation_id, handle_index = self._find_box_hit(image_pos)
        if annotation_id is not None:
            self.annotation_selected.emit(annotation_id)
            if self._mode == AnnotationMode.BOX:
                self._dragging_box_id = annotation_id
                self._dragging_corner_index = handle_index
                self._dragging_box_rect = self._box_rect_by_id(annotation_id)
                self._dragging_box_offset = None
                if handle_index is None and self._dragging_box_rect is not None:
                    left, top, _right, _bottom = self._dragging_box_rect
                    self._dragging_box_offset = (
                        image_pos[0] - left,
                        image_pos[1] - top,
                    )
            return

        if self._mode == AnnotationMode.POINT:
            self.point_created.emit(self.viewer_id, image_pos[0], image_pos[1])
            return
        if self._mode == AnnotationMode.BOX:
            self._drag_start = image_pos
            self._preview_rect = (
                image_pos[0],
                image_pos[1],
                image_pos[0],
                image_pos[1],
            )
            self.update()

    def mouseMoveEvent(self, event) -> None:
        image_pos = self._widget_to_image(event.position())
        if image_pos is None:
            return
        if self._dragging_box_id is not None and self._dragging_box_rect is not None:
            if self._dragging_corner_index is not None:
                rect = resize_rect_with_handle(
                    self._dragging_box_rect, self._dragging_corner_index, image_pos
                )
                self.box_updated.emit(self._dragging_box_id, rect)
                return
            if self._dragging_box_offset is not None:
                left, top, _right, _bottom = self._dragging_box_rect
                next_left = image_pos[0] - self._dragging_box_offset[0]
                next_top = image_pos[1] - self._dragging_box_offset[1]
                rect = move_rect(
                    self._dragging_box_rect,
                    (next_left - left, next_top - top),
                )
                self.box_updated.emit(self._dragging_box_id, rect)
                return
        if self._drag_start is not None and self._mode == AnnotationMode.BOX:
            self._preview_rect = (
                self._drag_start[0],
                self._drag_start[1],
                image_pos[0],
                image_pos[1],
            )
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        image_pos = self._widget_to_image(event.position())
        if self._dragging_box_id is not None:
            self._dragging_box_id = None
            self._dragging_corner_index = None
            self._dragging_box_rect = None
            self._dragging_box_offset = None
            return
        if (
            self._drag_start is None
            or image_pos is None
            or self._mode != AnnotationMode.BOX
        ):
            return
        left = min(self._drag_start[0], image_pos[0])
        top = min(self._drag_start[1], image_pos[1])
        right = max(self._drag_start[0], image_pos[0])
        bottom = max(self._drag_start[1], image_pos[1])
        if abs(right - left) >= 2 and abs(bottom - top) >= 2:
            self.box_created.emit(self.viewer_id, (left, top, right, bottom))
        self._drag_start = None
        self._preview_rect = None
        self.update()


class SliceViewWidget(QWidget):
    changed = pyqtSignal(str, str, object)
    point_created = pyqtSignal(str, float, float)
    box_created = pyqtSignal(str, tuple)
    box_updated = pyqtSignal(str, tuple)
    annotation_selected = pyqtSignal(str)

    def __init__(self, viewer_id: str, parent=None) -> None:
        super().__init__(parent)
        self.viewer_id = viewer_id
        self._pixmap: QPixmap | None = None
        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.orientation_combo = QComboBox()
        for orientation in SliceOrientation:
            self.orientation_combo.addItem(orientation.value.title(), orientation.value)
        self.index_slider = QSlider(Qt.Orientation.Horizontal)
        self.index_slider.setMinimum(0)
        self.index_slider.setMaximum(0)
        self.index_spinbox = QSpinBox()
        self.index_spinbox.setRange(0, 0)
        self.link_checkbox = QCheckBox("Linked")
        self.link_checkbox.setChecked(True)
        controls.addWidget(self.orientation_combo, 2)
        controls.addWidget(self.index_slider, 3)
        controls.addWidget(self.index_spinbox, 1)
        controls.addWidget(self.link_checkbox, 1)
        layout.addLayout(controls)

        overlays = QHBoxLayout()
        self.volume_checkbox = QCheckBox("Volume")
        self.volume_checkbox.setChecked(True)
        self.cam_checkbox = QCheckBox("CAM")
        self.cam_checkbox.setChecked(True)
        self.roi_checkbox = QCheckBox("ROI")
        self.roi_checkbox.setEnabled(False)
        self.mask_checkbox = QCheckBox("Mask")
        self.mask_checkbox.setEnabled(False)
        overlays.addWidget(self.volume_checkbox)
        overlays.addWidget(self.cam_checkbox)
        overlays.addWidget(self.roi_checkbox)
        overlays.addWidget(self.mask_checkbox)
        overlays.addStretch()
        layout.addLayout(overlays)

        self.canvas = SliceCanvasWidget(viewer_id)
        self.canvas.point_created.connect(self.point_created)
        self.canvas.box_created.connect(self.box_created)
        self.canvas.box_updated.connect(self.box_updated)
        self.canvas.annotation_selected.connect(self.annotation_selected)
        layout.addWidget(self.canvas, 1)

        self.orientation_combo.currentIndexChanged.connect(self._on_orientation_changed)
        self.index_slider.valueChanged.connect(self._on_slider_changed)
        self.index_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.link_checkbox.toggled.connect(self._on_link_toggled)
        self.volume_checkbox.toggled.connect(self._on_overlay_changed)
        self.cam_checkbox.toggled.connect(self._on_overlay_changed)

    def set_state(
        self, state: SliceViewState, volume_shape: tuple[int, int, int]
    ) -> None:
        self._building = True
        self.orientation_combo.setCurrentText(state.orientation.value.title())
        max_index = clamp_slice_index(state.orientation, 999999, volume_shape)
        self.index_slider.setMaximum(max_index)
        self.index_spinbox.setMaximum(max_index)
        self.index_slider.setValue(state.slice_index)
        self.index_spinbox.setValue(state.slice_index)
        self.link_checkbox.setChecked(state.is_linked)
        self.volume_checkbox.setChecked(state.overlay_modes.get("volume", True))
        self.cam_checkbox.setChecked(state.overlay_modes.get("cam", True))
        self._building = False

    def set_image(self, image: np.ndarray) -> None:
        height, width, _ = image.shape
        qimage = QImage(
            image.data, width, height, image.strides[0], QImage.Format.Format_RGB888
        )
        self._pixmap = QPixmap.fromImage(qimage.copy())

    def set_annotation_overlay(
        self,
        *,
        mode: AnnotationMode,
        point_size: int,
        selected_annotation_id: str | None,
        points: list[tuple[str, float, float, bool]],
        boxes: list[tuple[str, float, float, float, float, bool]],
        roi_projection: tuple[float, float, float, float] | None,
    ) -> None:
        image_size = (
            (self._pixmap.width(), self._pixmap.height()) if self._pixmap else (1, 1)
        )
        self.canvas.set_render_state(
            pixmap=self._pixmap,
            image_size=image_size,
            orientation=(
                SliceOrientation(self.orientation_combo.currentData())
                if self.orientation_combo.currentData()
                else SliceOrientation.AXIAL
            ),
            mode=mode,
            point_size=point_size,
            selected_annotation_id=selected_annotation_id,
            points=points,
            boxes=boxes,
            roi_projection=roi_projection,
        )

    def _on_orientation_changed(self) -> None:
        if self._building:
            return
        orientation = self.orientation_combo.currentData()
        self.changed.emit(self.viewer_id, "orientation", SliceOrientation(orientation))

    def _on_slider_changed(self, value: int) -> None:
        if self._building:
            return
        if self.index_spinbox.value() != value:
            self.index_spinbox.setValue(value)
        self.changed.emit(self.viewer_id, "slice_index", value)

    def _on_spinbox_changed(self, value: int) -> None:
        if self._building:
            return
        if self.index_slider.value() != value:
            self.index_slider.setValue(value)
        self.changed.emit(self.viewer_id, "slice_index", value)

    def _on_link_toggled(self, checked: bool) -> None:
        if self._building:
            return
        self.changed.emit(self.viewer_id, "is_linked", checked)

    def _on_overlay_changed(self) -> None:
        if self._building:
            return
        self.changed.emit(
            self.viewer_id,
            "overlay_modes",
            {
                "volume": self.volume_checkbox.isChecked(),
                "cam": self.cam_checkbox.isChecked(),
                "roi": self.roi_checkbox.isChecked(),
                "mask": self.mask_checkbox.isChecked(),
            },
        )


class ViewerWorkspace(QWidget):
    layout_changed = pyqtSignal(str, str)
    annotations_changed = pyqtSignal()
    volume_view_title = "3d view"

    def __init__(self, parent=None, *, enable_annotations: bool = True) -> None:
        super().__init__(parent)
        self.enable_annotations = enable_annotations
        self.state = WorkspaceState()
        self.presets = {preset.id: preset for preset in default_layout_presets()}
        self.payload = ViewerPayload()
        self.overlay_status_messages: list[str] = []
        self.tile_widgets: dict[str, ViewerTileWidget] = {}
        self.slice_widgets: dict[str, SliceViewWidget] = {}
        self.viewer_slice_states: dict[str, SliceViewState] = {}
        self.slot_hosts: dict[str, SlotHost] = {}
        self.viewer_to_slot: dict[str, str] = {}
        self.splitters: dict[str, QSplitter] = {}
        self.current_preset = self.presets[self.state.active_layout_id]
        self.layout_root_widget: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.workspace_frame = QFrame()
        self.workspace_frame.setObjectName("workspaceCanvas")
        self.workspace_layout = QVBoxLayout(self.workspace_frame)
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.workspace_frame, 1)

        self._create_viewers()
        if self.enable_annotations:
            self.renderer.set_annotation_event_handler(
                self._handle_renderer_annotation_event
            )
        self.apply_layout(self.state.active_layout_id)

    def _create_viewers(self) -> None:
        self._create_3d_viewer()
        for viewer_id in ("slice-1", "slice-2", "slice-3"):
            self._create_slice_viewer(viewer_id)

    def _create_3d_viewer(self) -> None:
        self.volume_content = QWidget()
        content_layout = QVBoxLayout(self.volume_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.vtk_widget = QVTKRenderWindowInteractor(self.volume_content)
        self.vtk_widget.setObjectName("volumeViewport")
        content_layout.addWidget(self.vtk_widget, 1)
        self.renderer = self._create_renderer(self.vtk_widget)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.state.tiles["viewer-3d"] = ViewerTileState(
            "viewer-3d", ViewerType.VOLUME_3D, self.volume_view_title
        )
        tile = ViewerTileWidget("viewer-3d", self.volume_view_title)
        tile.set_content(self.volume_content)
        tile.selected.connect(self._on_viewer_selected)
        self.tile_widgets["viewer-3d"] = tile

    def _create_renderer(self, vtk_widget):
        return StandardMultiVolumeRenderer(vtk_widget)

    def _create_slice_viewer(self, viewer_id: str) -> None:
        widget = SliceViewWidget(viewer_id)
        widget.changed.connect(self._on_slice_widget_changed)
        widget.point_created.connect(self._add_point_from_slice)
        widget.box_created.connect(self._add_box_from_slice)
        widget.box_updated.connect(self._update_box_from_slice)
        widget.annotation_selected.connect(self.select_annotation)
        self.slice_widgets[viewer_id] = widget
        self.viewer_slice_states[viewer_id] = SliceViewState(
            viewer_id, SliceOrientation.AXIAL
        )
        self.state.tiles[viewer_id] = ViewerTileState(
            viewer_id, ViewerType.SLICE_2D, "Slice"
        )
        tile = ViewerTileWidget(viewer_id, "Slice")
        tile.set_content(widget)
        tile.selected.connect(self._on_viewer_selected)
        self.tile_widgets[viewer_id] = tile

    def apply_layout(self, preset_id: str) -> None:
        preset = self.presets[preset_id]
        self.current_preset = preset
        self.state.active_layout_id = preset_id
        self._ensure_slot_states(preset)
        self._initialize_default_assignments(preset)
        self._rebuild_layout_tree(preset)
        self._normalize_slot_states()
        self.refresh_slice_views()
        self._refresh_3d_view()
        self.layout_changed.emit(preset.id, preset.title)

    def _ensure_slot_states(self, preset: LayoutPreset) -> None:
        layout_states = self.state.slot_slice_states.setdefault(preset.id, {})
        for slot in collect_slots(preset.root_node):
            if slot.viewer_type != ViewerType.SLICE_2D:
                continue
            layout_states.setdefault(
                slot.slot_id,
                SliceViewState(
                    id=slot.slot_id,
                    orientation=slot.default_orientation or SliceOrientation.AXIAL,
                ),
            )

    @staticmethod
    def _copy_slice_state(source: SliceViewState, target_id: str) -> SliceViewState:
        return SliceViewState(
            id=target_id,
            orientation=source.orientation,
            slice_index=source.slice_index,
            is_linked=source.is_linked,
            overlay_modes=dict(source.overlay_modes),
        )

    def _reset_slot_states(self, preset: LayoutPreset) -> None:
        self.state.slot_slice_states[preset.id] = {
            slot.slot_id: SliceViewState(
                id=slot.slot_id,
                orientation=slot.default_orientation or SliceOrientation.AXIAL,
            )
            for slot in collect_slots(preset.root_node)
            if slot.viewer_type == ViewerType.SLICE_2D
        }

    def _initialize_default_assignments(self, preset: LayoutPreset) -> None:
        assignments: dict[str, str] = {}
        slice_viewers = ["slice-1", "slice-2", "slice-3"]
        slice_index = 0
        for slot in collect_slots(preset.root_node):
            if slot.viewer_type == ViewerType.VOLUME_3D:
                assignments[slot.slot_id] = "viewer-3d"
            else:
                viewer_id = slice_viewers[slice_index]
                assignments[slot.slot_id] = viewer_id
                slot_state = self.state.slot_slice_states[preset.id][slot.slot_id]
                self.viewer_slice_states[viewer_id] = self._copy_slice_state(
                    slot_state, viewer_id
                )
                slice_index += 1
        self.state.slot_assignments = assignments
        self.viewer_to_slot = {
            viewer_id: slot_id for slot_id, viewer_id in assignments.items()
        }
        for viewer_id, tile_state in self.state.tiles.items():
            tile_state.is_visible = viewer_id in self.viewer_to_slot

    def _rebuild_layout_tree(self, preset: LayoutPreset) -> None:
        for tile in self.tile_widgets.values():
            tile.setParent(None)
        self.slot_hosts = {}
        self.splitters = {}
        while self.workspace_layout.count():
            item = self.workspace_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
                child.deleteLater()
        self.layout_root_widget = None
        self.layout_root_widget = self._build_node_widget(preset.root_node)
        self.workspace_layout.addWidget(self.layout_root_widget, 1)
        for slot_id, viewer_id in self.state.slot_assignments.items():
            self._attach_viewer_to_slot(slot_id, viewer_id)
        self._restore_splitter_sizes()

    def _build_node_widget(self, node: LayoutNode) -> QWidget:
        splitter = QSplitter(_to_qt_orientation(node.orientation))
        splitter.setChildrenCollapsible(False)
        splitter.setObjectName("workspaceSplitter")
        splitter.splitterMoved.connect(
            lambda _pos, _index, node_id=node.node_id: self._remember_splitter_sizes(
                node_id
            )
        )
        self.splitters[node.node_id] = splitter
        for child in node.children:
            if isinstance(child, LayoutSlot):
                host = SlotHost(child)
                host.swap_requested.connect(self.swap_viewers)
                self.slot_hosts[child.slot_id] = host
                splitter.addWidget(host)
            else:
                splitter.addWidget(self._build_node_widget(child))
        splitter.setSizes(list(node.sizes))
        return splitter

    def _restore_splitter_sizes(self) -> None:
        stored = self.state.splitter_sizes.get(self.state.active_layout_id, {})
        for node_id, splitter in self.splitters.items():
            if node_id in stored:
                splitter.setSizes(stored[node_id])

    def _remember_splitter_sizes(self, node_id: str) -> None:
        layout_sizes = self.state.splitter_sizes.setdefault(
            self.state.active_layout_id, {}
        )
        layout_sizes[node_id] = self.splitters[node_id].sizes()

    def _attach_viewer_to_slot(self, slot_id: str, viewer_id: str) -> None:
        host = self.slot_hosts[slot_id]
        host.bind_tile(viewer_id, self.tile_widgets[viewer_id])
        self.viewer_to_slot[viewer_id] = slot_id
        self.state.tiles[viewer_id].is_visible = True

    def swap_viewers(self, source_viewer_id: str, target_viewer_id: str) -> None:
        if source_viewer_id == target_viewer_id:
            return
        source_slot_id = self.viewer_to_slot.get(source_viewer_id)
        target_slot_id = self.viewer_to_slot.get(target_viewer_id)
        if source_slot_id is None or target_slot_id is None:
            return
        self.state.slot_assignments[source_slot_id] = target_viewer_id
        self.state.slot_assignments[target_slot_id] = source_viewer_id
        self._attach_viewer_to_slot(source_slot_id, target_viewer_id)
        self._attach_viewer_to_slot(target_slot_id, source_viewer_id)
        self.refresh_slice_views()

    def _on_viewer_selected(self, viewer_id: str) -> None:
        self.state.active_viewer_id = viewer_id

    def _slot_state_for_viewer(
        self, viewer_id: str
    ) -> tuple[str | None, SliceViewState | None]:
        slot_id = self.viewer_to_slot.get(viewer_id)
        state = self.viewer_slice_states.get(viewer_id)
        return slot_id, state

    def capture_slice_snapshot(self) -> dict[str, tuple[SliceOrientation, int]]:
        return {
            viewer_id: (state.orientation, state.slice_index)
            for viewer_id, state in self.viewer_slice_states.items()
        }

    def apply_slice_snapshot(
        self, snapshot: dict[str, tuple[SliceOrientation, int]] | None
    ) -> None:
        if not snapshot:
            return
        for viewer_id, values in snapshot.items():
            state = self.viewer_slice_states.get(viewer_id)
            if state is None:
                continue
            orientation, slice_index = values
            state.orientation = orientation
            state.slice_index = clamp_slice_index(
                orientation, slice_index, self.state.volume_shape
            )
        self.refresh_slice_views()

    def set_workspace_payload(
        self,
        *,
        renderable_items: list[dict[str, object]],
    ) -> None:
        normalized_items = []
        for item in renderable_items:
            normalized_items.append(
                {
                    **item,
                    "data": _as_numpy(item.get("data")),
                }
            )
        self.payload = ViewerPayload(renderable_items=normalized_items)
        self.state.volume_shape = self._reference_volume_shape()
        self._normalize_slot_states()
        self.refresh_slice_views()
        self._refresh_renderer_annotations()

    def refresh_slice_views(self) -> None:
        mode = (
            self.state.annotations.mode
            if self.enable_annotations
            else AnnotationMode.OFF
        )
        point_size = self.state.annotations.point_size
        self.overlay_status_messages = []
        visible_base_items = [
            item
            for item in self.payload.renderable_items
            if item.get("source") == "base"
        ]
        visible_xai_items = [
            item
            for item in self.payload.renderable_items
            if item.get("source") == "xai" and bool(item.get("visible", True))
        ]
        base_item = self._select_reference_base_item(
            visible_base_items, visible_xai_items
        )
        self.state.volume_shape = (
            tuple(int(v) for v in base_item["data"].shape)
            if base_item is not None and base_item.get("data") is not None
            else (0, 0, 0)
        )
        self._normalize_slot_states()
        for viewer_id, widget in self.slice_widgets.items():
            slot_id = self.viewer_to_slot.get(viewer_id)
            if slot_id is None:
                self.tile_widgets[viewer_id].hide()
                continue
            state = self.viewer_slice_states[viewer_id]
            widget.set_state(state, self.state.volume_shape)
            volume_slice = None
            if (
                base_item is not None
                and state.overlay_modes.get("volume", True)
                and base_item.get("data") is not None
            ):
                index = clamp_slice_index(
                    state.orientation, state.slice_index, self.state.volume_shape
                )
                volume_slice = _extract_slice(
                    base_item["data"], state.orientation, index
                )
            xai_slices: list[tuple[np.ndarray, np.ndarray]] = []
            if state.overlay_modes.get("cam", True):
                index = clamp_slice_index(
                    state.orientation, state.slice_index, self.state.volume_shape
                )
                for item in visible_xai_items:
                    if item.get("data") is None or base_item is None:
                        continue
                    xai_slice, status_message = _resample_item_slice_to_base(
                        base_item, item, state.orientation, index
                    )
                    if (
                        status_message
                        and status_message not in self.overlay_status_messages
                    ):
                        self.overlay_status_messages.append(status_message)
                    if xai_slice is None:
                        continue
                    xai_slices.append(
                        (
                            xai_slice,
                            _color_map_from_transfer_function(
                                item["transfer_function"], item["data_range"]
                            ),
                        )
                    )
            widget.set_image(_blend_slice_image(volume_slice, xai_slices, state))
            points = (
                self._slice_points_for_viewer(viewer_id, state)
                if self.enable_annotations
                else []
            )
            boxes = (
                self._slice_boxes_for_viewer(viewer_id, state)
                if self.enable_annotations
                else []
            )
            roi_projection = (
                self._roi_projection_for_viewer(state)
                if self.enable_annotations
                else None
            )
            widget.set_annotation_overlay(
                mode=mode,
                point_size=point_size,
                selected_annotation_id=self.state.annotations.selected_annotation_id,
                points=points,
                boxes=boxes,
                roi_projection=roi_projection,
            )
            self.tile_widgets[viewer_id].set_title(
                f"{state.orientation.value.title()} Slice"
            )
            self.tile_widgets[viewer_id].show()
        self.tile_widgets["viewer-3d"].set_title(self.volume_view_title)
        self._refresh_renderer_annotations()
        self._refresh_3d_view()

    def set_slice_orientation(
        self,
        viewer_id: str,
        orientation: SliceOrientation,
        *,
        preserve_link: bool = True,
    ) -> None:
        slot_id, state = self._slot_state_for_viewer(viewer_id)
        if slot_id is None or state is None:
            return
        state.orientation = orientation
        state.slice_index = clamp_slice_index(
            orientation, state.slice_index, self.state.volume_shape
        )
        if not preserve_link:
            state.is_linked = False
        layout_states = self.state.slot_slice_states.get(
            self.state.active_layout_id, {}
        )
        if slot_id in layout_states:
            layout_states[slot_id] = self._copy_slice_state(state, slot_id)
        self.refresh_slice_views()

    def set_slice_index(self, viewer_id: str, index: int) -> None:
        slot_id, state = self._slot_state_for_viewer(viewer_id)
        if slot_id is None or state is None:
            return
        state.slice_index = clamp_slice_index(
            state.orientation, index, self.state.volume_shape
        )
        if self.state.global_slice_link_mode and state.is_linked:
            for other_viewer_id, other_state in self.viewer_slice_states.items():
                if other_viewer_id == viewer_id or not other_state.is_linked:
                    continue
                if other_state.orientation == state.orientation:
                    other_state.slice_index = clamp_slice_index(
                        other_state.orientation,
                        state.slice_index,
                        self.state.volume_shape,
                    )
                    other_slot = self.viewer_to_slot.get(other_viewer_id)
                    layout_states = self.state.slot_slice_states.get(
                        self.state.active_layout_id, {}
                    )
                    if other_slot in layout_states:
                        layout_states[other_slot] = self._copy_slice_state(
                            other_state, other_slot
                        )
        layout_states = self.state.slot_slice_states.get(
            self.state.active_layout_id, {}
        )
        if slot_id in layout_states:
            layout_states[slot_id] = self._copy_slice_state(state, slot_id)
        self.refresh_slice_views()

    def _on_slice_widget_changed(self, viewer_id: str, field: str, value) -> None:
        slot_id, state = self._slot_state_for_viewer(viewer_id)
        if slot_id is None or state is None:
            return
        if field == "orientation":
            self.set_slice_orientation(viewer_id, value)
            return
        if field == "slice_index":
            self.set_slice_index(viewer_id, value)
            return
        if field == "is_linked":
            state.is_linked = bool(value)
        elif field == "overlay_modes":
            state.overlay_modes = dict(value)
        layout_states = self.state.slot_slice_states.get(
            self.state.active_layout_id, {}
        )
        if slot_id in layout_states:
            layout_states[slot_id] = self._copy_slice_state(state, slot_id)
        self.refresh_slice_views()

    def _normalize_slot_states(self) -> None:
        if not any(self.state.volume_shape):
            return
        center = tuple(max(0, dimension // 2) for dimension in self.state.volume_shape)
        self.state.crosshair_position = center
        for state in self.state.slot_slice_states.get(
            self.state.active_layout_id, {}
        ).values():
            axis = orientation_axis(state.orientation)
            if state.slice_index == 0:
                state.slice_index = center[axis]
            state.slice_index = clamp_slice_index(
                state.orientation, state.slice_index, self.state.volume_shape
            )

    def _refresh_3d_view(self) -> None:
        if not hasattr(self, "vtk_widget"):
            return
        self.volume_content.show()
        self.vtk_widget.show()
        self.volume_content.updateGeometry()
        self.vtk_widget.updateGeometry()
        self.vtk_widget.update()
        self._refresh_3d_tile_chrome()
        QTimer.singleShot(0, self._refresh_3d_tile_chrome)
        QTimer.singleShot(25, self._refresh_3d_tile_chrome)
        QTimer.singleShot(100, self._refresh_3d_tile_chrome)
        QTimer.singleShot(0, self.renderer.render)
        QTimer.singleShot(25, self.renderer.render)

    def _refresh_3d_tile_chrome(self) -> None:
        tile = self.tile_widgets.get("viewer-3d")
        if tile is None:
            return
        tile.set_title(self.volume_view_title)
        tile.show()
        tile.header.show()
        tile.header.title_label.show()
        tile.header.raise_()
        tile.header.title_label.raise_()
        tile.updateGeometry()
        tile.header.updateGeometry()
        tile.header.title_label.updateGeometry()
        tile.update()
        tile.header.update()
        tile.header.title_label.update()

    def shutdown(self) -> None:
        if hasattr(self, "renderer"):
            self.renderer.shutdown()

    def _refresh_renderer_annotations(self) -> None:
        if not hasattr(self, "renderer"):
            return
        if not self.enable_annotations:
            self.renderer.set_annotation_mode(AnnotationMode.OFF.value)
            self.renderer.set_annotations(
                [], [], None, None, self.state.annotations.point_size
            )
            return
        self.renderer.set_annotation_mode(self.state.annotations.mode.value)
        self.renderer.set_annotations(
            self.state.annotations.points,
            self.state.annotations.boxes_3d,
            self.state.annotations.selected_annotation_id,
            self.state.annotations.active_roi_box_id,
            self.state.annotations.point_size,
        )
        self.annotations_changed.emit()

    def set_annotation_mode(self, mode: str | AnnotationMode) -> None:
        self.state.annotations.mode = (
            mode if isinstance(mode, AnnotationMode) else AnnotationMode(mode)
        )
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def set_annotation_point_size(self, size: int) -> None:
        self.state.annotations.point_size = max(2, int(size))
        for item in self.state.annotations.points:
            item.size = self.state.annotations.point_size
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def set_active_roi_box(self, box_id: str | None) -> None:
        self.state.annotations.active_roi_box_id = box_id
        for box in self.state.annotations.boxes_3d:
            box.is_roi_target = box.id == box_id
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def select_annotation(self, annotation_id: str | None) -> None:
        self.state.annotations.selected_annotation_id = annotation_id
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def export_annotations(self):
        return deepcopy(self.state.annotations)

    def import_annotations(self, state) -> None:
        self.state.annotations = deepcopy(state)
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def clear_annotations(self) -> None:
        self.state.annotations = WorkspaceState().annotations
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def delete_selected_annotation(self) -> None:
        annotation_id = self.state.annotations.selected_annotation_id
        if not annotation_id:
            return
        self.state.annotations.points = [
            item for item in self.state.annotations.points if item.id != annotation_id
        ]
        self.state.annotations.boxes_2d = [
            item for item in self.state.annotations.boxes_2d if item.id != annotation_id
        ]
        self.state.annotations.boxes_3d = [
            item for item in self.state.annotations.boxes_3d if item.id != annotation_id
        ]
        if self.state.annotations.active_roi_box_id == annotation_id:
            self.state.annotations.active_roi_box_id = None
        self.state.annotations.selected_annotation_id = None
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def _slice_points_for_viewer(self, viewer_id: str, state: SliceViewState):
        results = []
        for item in self.state.annotations.points:
            point = self._voxel_to_slice_coords(item.position, state)
            if point is None:
                continue
            results.append(
                (
                    item.id,
                    point[0],
                    point[1],
                    item.id == self.state.annotations.selected_annotation_id,
                )
            )
        return results

    def _slice_boxes_for_viewer(self, viewer_id: str, state: SliceViewState):
        results = []
        for item in self.state.annotations.boxes_2d:
            if (
                item.orientation != state.orientation
                or item.slice_index != state.slice_index
            ):
                continue
            results.append(
                (
                    item.id,
                    item.rect[0],
                    item.rect[1],
                    item.rect[2],
                    item.rect[3],
                    item.id == self.state.annotations.selected_annotation_id,
                )
            )
        return results

    def _roi_projection_for_viewer(self, state: SliceViewState):
        active_id = self.state.annotations.active_roi_box_id
        if not active_id:
            return None
        target = next(
            (item for item in self.state.annotations.boxes_3d if item.id == active_id),
            None,
        )
        if target is None:
            return None
        min_corner = target.min_corner
        max_corner = target.max_corner
        volume_shape = self.state.volume_shape
        max_axis0 = max(0, volume_shape[0] - 1)
        max_axis1 = max(0, volume_shape[1] - 1)
        max_axis2 = max(0, volume_shape[2] - 1)
        if state.orientation == SliceOrientation.AXIAL:
            if not (min_corner[1] <= state.slice_index <= max_corner[1]):
                return None
            return (
                max_axis2 - max_corner[2],
                max_axis0 - max_corner[0],
                max_axis2 - min_corner[2],
                max_axis0 - min_corner[0],
            )
        if state.orientation == SliceOrientation.CORONAL:
            if not (min_corner[2] <= state.slice_index <= max_corner[2]):
                return None
            return (
                max_axis0 - max_corner[0],
                max_axis1 - max_corner[1],
                max_axis0 - min_corner[0],
                max_axis1 - min_corner[1],
            )
        if not (min_corner[0] <= state.slice_index <= max_corner[0]):
            return None
        return (
            max_axis2 - max_corner[2],
            max_axis1 - max_corner[1],
            max_axis2 - min_corner[2],
            max_axis1 - min_corner[1],
        )

    def _voxel_to_slice_coords(
        self, position: tuple[float, float, float], state: SliceViewState
    ):
        max_axis0 = max(0, self.state.volume_shape[0] - 1)
        max_axis1 = max(0, self.state.volume_shape[1] - 1)
        max_axis2 = max(0, self.state.volume_shape[2] - 1)
        if state.orientation == SliceOrientation.AXIAL:
            if int(round(position[1])) != state.slice_index:
                return None
            return (max_axis2 - position[2], max_axis0 - position[0])
        if state.orientation == SliceOrientation.CORONAL:
            if int(round(position[2])) != state.slice_index:
                return None
            return (max_axis0 - position[0], max_axis1 - position[1])
        if int(round(position[0])) != state.slice_index:
            return None
        return (max_axis2 - position[2], max_axis1 - position[1])

    def _slice_to_voxel_coords(self, state: SliceViewState, x: float, y: float):
        max_axis0 = max(0, self.state.volume_shape[0] - 1)
        max_axis1 = max(0, self.state.volume_shape[1] - 1)
        max_axis2 = max(0, self.state.volume_shape[2] - 1)
        if state.orientation == SliceOrientation.AXIAL:
            return (
                max_axis0 - y,
                float(state.slice_index),
                max_axis2 - x,
            )
        if state.orientation == SliceOrientation.CORONAL:
            return (
                max_axis0 - x,
                max_axis1 - y,
                float(state.slice_index),
            )
        return (
            float(state.slice_index),
            max_axis1 - y,
            max_axis2 - x,
        )

    def _handle_renderer_annotation_event(self, event_type: str, payload: dict) -> None:
        if event_type == "add_point_3d":
            point = PointAnnotation(
                id=f"pt-{uuid4().hex[:8]}",
                space="voxel",
                position=tuple(float(v) for v in payload["position"]),
                size=self.state.annotations.point_size,
                source_viewer_id="viewer-3d",
            )
            self.state.annotations.points.append(point)
            self.state.annotations.selected_annotation_id = point.id
        elif event_type == "add_box_3d":
            start = payload["min_corner"]
            end = payload["max_corner"]
            box = Box3DAnnotation(
                id=f"box3d-{uuid4().hex[:8]}",
                min_corner=tuple(min(float(start[i]), float(end[i])) for i in range(3)),
                max_corner=tuple(max(float(start[i]), float(end[i])) for i in range(3)),
            )
            self.state.annotations.boxes_3d.append(box)
            self.state.annotations.selected_annotation_id = box.id
        elif event_type == "resize_box_3d":
            self._resize_3d_box(
                payload["annotation_id"], payload["corner_index"], payload["position"]
            )
        elif event_type == "move_box_3d":
            self._move_3d_box(
                payload["annotation_id"],
                payload["delta"],
                payload["initial_min_corner"],
                payload["initial_max_corner"],
            )
        elif event_type == "select_annotation":
            self.state.annotations.selected_annotation_id = payload["annotation_id"]
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def overlay_status_message(self) -> str:
        unique_messages = list(dict.fromkeys(self.overlay_status_messages))
        return " | ".join(unique_messages[:3])

    def _reference_volume_shape(self) -> tuple[int, int, int]:
        for item in self.payload.renderable_items:
            data = item.get("data")
            if item.get("source") == "base" and data is not None:
                return tuple(int(v) for v in data.shape)
        for item in self.payload.renderable_items:
            data = item.get("data")
            if data is not None:
                return tuple(int(v) for v in data.shape)
        return (0, 0, 0)

    def _select_reference_base_item(
        self,
        base_items: list[dict[str, object]],
        xai_items: list[dict[str, object]],
    ) -> dict[str, object] | None:
        focused_item = next(
            (
                item
                for item in self.payload.renderable_items
                if bool(item.get("is_focus"))
            ),
            None,
        )
        if isinstance(focused_item, dict):
            if (
                focused_item.get("source") == "base"
                and focused_item.get("data") is not None
            ):
                return focused_item
            source_base_id = focused_item.get("source_base_item_id")
            if source_base_id:
                for item in base_items:
                    if (
                        item.get("id") == source_base_id
                        and item.get("data") is not None
                    ):
                        return item
        visible_base_items = [
            item for item in base_items if bool(item.get("visible", True))
        ]
        if visible_base_items:
            return visible_base_items[0]
        if xai_items:
            source_base_id = xai_items[0].get("source_base_item_id")
            if source_base_id:
                for item in base_items:
                    if (
                        item.get("id") == source_base_id
                        and item.get("data") is not None
                    ):
                        return item
        return base_items[0] if base_items else None

    def _resize_3d_box(self, annotation_id: str, corner_index: int, position) -> None:
        for box in self.state.annotations.boxes_3d:
            if box.id != annotation_id:
                continue
            corners = [
                [box.min_corner[0], box.min_corner[1], box.min_corner[2]],
                [box.max_corner[0], box.min_corner[1], box.min_corner[2]],
                [box.min_corner[0], box.max_corner[1], box.min_corner[2]],
                [box.max_corner[0], box.max_corner[1], box.min_corner[2]],
                [box.min_corner[0], box.min_corner[1], box.max_corner[2]],
                [box.max_corner[0], box.min_corner[1], box.max_corner[2]],
                [box.min_corner[0], box.max_corner[1], box.max_corner[2]],
                [box.max_corner[0], box.max_corner[1], box.max_corner[2]],
            ]
            corners[corner_index] = [float(v) for v in position]
            box.min_corner = tuple(min(c[i] for c in corners) for i in range(3))
            box.max_corner = tuple(max(c[i] for c in corners) for i in range(3))
            break

    def _move_3d_box(
        self, annotation_id: str, delta, initial_min_corner, initial_max_corner
    ) -> None:
        for box in self.state.annotations.boxes_3d:
            if box.id != annotation_id:
                continue
            next_min = tuple(
                float(initial_min_corner[i]) + float(delta[i]) for i in range(3)
            )
            next_max = tuple(
                float(initial_max_corner[i]) + float(delta[i]) for i in range(3)
            )
            clamped_min, clamped_max = self._clamp_3d_box_bounds(next_min, next_max)
            box.min_corner = clamped_min
            box.max_corner = clamped_max
            break

    def _clamp_3d_box_bounds(self, min_corner, max_corner):
        volume_shape = self.state.volume_shape
        if not any(volume_shape):
            return tuple(float(v) for v in min_corner), tuple(
                float(v) for v in max_corner
            )
        lengths = [float(max_corner[i]) - float(min_corner[i]) for i in range(3)]
        clamped_min = []
        clamped_max = []
        for axis in range(3):
            axis_limit = float(volume_shape[axis] - 1)
            min_value = float(min_corner[axis])
            max_value = float(max_corner[axis])
            if min_value < 0.0:
                max_value -= min_value
                min_value = 0.0
            if max_value > axis_limit:
                min_value -= max_value - axis_limit
                max_value = axis_limit
            if lengths[axis] > axis_limit:
                min_value = 0.0
                max_value = axis_limit
            min_value = max(0.0, min(min_value, axis_limit))
            max_value = max(0.0, min(max_value, axis_limit))
            clamped_min.append(min_value)
            clamped_max.append(max_value)
        return tuple(clamped_min), tuple(clamped_max)

    def _add_point_from_slice(self, viewer_id: str, x: float, y: float) -> None:
        state = self.viewer_slice_states[viewer_id]
        point = PointAnnotation(
            id=f"pt-{uuid4().hex[:8]}",
            space="voxel",
            position=self._slice_to_voxel_coords(state, x, y),
            size=self.state.annotations.point_size,
            source_viewer_id=viewer_id,
        )
        self.state.annotations.points.append(point)
        self.state.annotations.selected_annotation_id = point.id
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def _add_box_from_slice(
        self, viewer_id: str, rect: tuple[float, float, float, float]
    ) -> None:
        state = self.viewer_slice_states[viewer_id]
        box = Box2DAnnotation(
            id=f"box2d-{uuid4().hex[:8]}",
            orientation=state.orientation,
            slice_index=state.slice_index,
            rect=tuple(float(v) for v in rect),
            source_viewer_id=viewer_id,
        )
        self.state.annotations.boxes_2d.append(box)
        self.state.annotations.selected_annotation_id = box.id
        self._refresh_renderer_annotations()
        self.refresh_slice_views()

    def _update_box_from_slice(
        self, annotation_id: str, rect: tuple[float, float, float, float]
    ) -> None:
        for box in self.state.annotations.boxes_2d:
            if box.id == annotation_id:
                box.rect = tuple(float(v) for v in rect)
                self.state.annotations.selected_annotation_id = box.id
                break
        self._refresh_renderer_annotations()
        self.refresh_slice_views()


class StandardWorkspace(ViewerWorkspace):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, enable_annotations=False)


class RoiWorkspace(ViewerWorkspace):
    volume_view_title = "Interactive 3D"

    def __init__(self, parent=None) -> None:
        super().__init__(parent, enable_annotations=True)

    def _create_renderer(self, vtk_widget):
        return VtkVolumeRenderer(vtk_widget)


class WorkspaceHost(QWidget):
    layout_changed = pyqtSignal(str, str)
    annotations_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mode = WorkspaceMode.STANDARD
        self.shared_state = SharedImagingState()
        self._scene_initialized = False
        self._scene_signature: tuple[tuple[str, tuple[int, ...]], ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        from PyQt6.QtWidgets import QStackedWidget

        self.stack = QStackedWidget(self)
        self.standard_workspace = StandardWorkspace(self)
        self.roi_workspace = RoiWorkspace(self)
        self.stack.addWidget(self.standard_workspace)
        self.stack.addWidget(self.roi_workspace)
        layout.addWidget(self.stack, 1)

        self.standard_workspace.layout_changed.connect(self.layout_changed.emit)
        self.roi_workspace.layout_changed.connect(self.layout_changed.emit)
        self.roi_workspace.annotations_changed.connect(self.annotations_changed.emit)

        self.stack.setCurrentWidget(self.standard_workspace)

    @property
    def active_workspace(self) -> ViewerWorkspace:
        return (
            self.roi_workspace
            if self.mode == WorkspaceMode.ROI
            else self.standard_workspace
        )

    @property
    def renderer(self):
        return self.active_workspace.renderer

    @property
    def rotating(self) -> bool:
        return bool(self.active_workspace.renderer.rotating)

    def set_workspace_mode(self, mode: str | WorkspaceMode) -> None:
        target_mode = mode if isinstance(mode, WorkspaceMode) else WorkspaceMode(mode)
        if target_mode == self.mode:
            return
        current = self.active_workspace
        was_rotating = current.renderer.rotating
        current.renderer.stop_rotation()
        self.shared_state.camera_snapshot = current.renderer.capture_camera_state()
        self.shared_state.slice_snapshot = current.capture_slice_snapshot()

        self.mode = target_mode
        if self.mode == WorkspaceMode.ROI:
            self.stack.setCurrentWidget(self.roi_workspace)
            if (
                self.roi_workspace.current_preset.id
                != self.standard_workspace.current_preset.id
            ):
                self.roi_workspace.apply_layout(self.standard_workspace.current_preset.id)
            self._apply_shared_snapshot_to(self.roi_workspace)
        else:
            self.stack.setCurrentWidget(self.standard_workspace)
            self._apply_shared_snapshot_to(self.standard_workspace)
        if was_rotating:
            self.active_workspace.renderer.start_rotation()
        self.active_workspace.renderer.render()

    def _apply_shared_snapshot_to(self, workspace: ViewerWorkspace) -> None:
        workspace.apply_slice_snapshot(self.shared_state.slice_snapshot)
        workspace.renderer.apply_camera_state(self.shared_state.camera_snapshot)

    def apply_layout(self, preset_id: str) -> None:
        self.standard_workspace.apply_layout(preset_id)
        self.roi_workspace.apply_layout(preset_id)
        if self.mode == WorkspaceMode.STANDARD:
            self.layout_changed.emit(
                self.standard_workspace.current_preset.id,
                self.standard_workspace.current_preset.title,
            )

    def set_workspace_payload(
        self,
        *,
        renderable_items: list[dict[str, object]],
    ) -> None:
        self.shared_state.renderable_items = list(renderable_items)
        self.standard_workspace.set_workspace_payload(
            renderable_items=renderable_items,
        )
        self.roi_workspace.set_workspace_payload(
            renderable_items=renderable_items,
        )

    def set_annotation_mode(self, mode: str | AnnotationMode) -> None:
        self.roi_workspace.set_annotation_mode(mode)

    def set_annotation_point_size(self, size: int) -> None:
        self.roi_workspace.set_annotation_point_size(size)

    def set_active_roi_box(self, box_id: str | None) -> None:
        self.roi_workspace.set_active_roi_box(box_id)

    def export_annotations(self):
        return self.roi_workspace.export_annotations()

    def import_annotations(self, state) -> None:
        self.roi_workspace.import_annotations(state)

    def clear_annotations(self) -> None:
        self.roi_workspace.clear_annotations()

    def delete_selected_annotation(self) -> None:
        self.roi_workspace.delete_selected_annotation()

    def select_annotation(self, annotation_id: str | None) -> None:
        self.roi_workspace.select_annotation(annotation_id)

    def show_volumes(
        self,
        volumes: list[object],
        spacing: list[tuple[float, float, float]],
        metadata: list[dict[str, object] | None] | None = None,
    ) -> None:
        metadata_items = metadata if metadata is not None else [None] * len(volumes)
        scene_signature = tuple(
            (
                str((meta or {}).get("volume_id", index)),
                tuple(int(v) for v in np.array(volume).shape),
            )
            for index, (volume, meta) in enumerate(zip(volumes, metadata_items))
        )
        self.standard_workspace.renderer.show_volumes(volumes, spacing, metadata)
        self.roi_workspace.renderer.show_volumes(volumes, spacing, metadata)
        scene_changed = (
            not self._scene_initialized
        ) or scene_signature != self._scene_signature
        if scene_changed:
            self.sync_camera_to_visible_volumes()
            self._scene_initialized = True
            self._scene_signature = scene_signature
        elif self.shared_state.camera_snapshot:
            self._apply_shared_snapshot_to(self.standard_workspace)
            self._apply_shared_snapshot_to(self.roi_workspace)
        self.standard_workspace.renderer.render()
        self.roi_workspace.renderer.render()

    def set_volume_transfer_functions(
        self,
        index: int,
        color_points: list[tuple[float, float, float, float]],
        opacity_points: list[tuple[float, float]],
        *,
        visible: bool | None = None,
        render: bool = True,
    ) -> None:
        self.standard_workspace.renderer.set_volume_transfer_functions(
            index, color_points, opacity_points, visible=visible, render=False
        )
        self.roi_workspace.renderer.set_volume_transfer_functions(
            index, color_points, opacity_points, visible=visible, render=False
        )
        if render:
            self.standard_workspace.renderer.render()
            self.roi_workspace.renderer.render()

    def set_rotation_speed(self, speed: float) -> None:
        self.standard_workspace.renderer.set_rotation_speed(speed)
        self.roi_workspace.renderer.set_rotation_speed(speed)

    def start_rotation(self) -> None:
        self.active_workspace.renderer.start_rotation()

    def stop_rotation(self) -> None:
        self.active_workspace.renderer.stop_rotation()

    def clear_volumes(self) -> None:
        self.standard_workspace.renderer.clear_volumes()
        self.roi_workspace.renderer.clear_volumes()
        self._scene_initialized = False
        self._scene_signature = ()

    def render(self) -> None:
        self.standard_workspace.renderer.render()
        self.roi_workspace.renderer.render()

    def overlay_status_message(self) -> str:
        return self.active_workspace.overlay_status_message()

    def store_initial_camera(self) -> None:
        self.standard_workspace.renderer.store_initial_camera()
        self.roi_workspace.renderer.store_initial_camera()

    def replace_camera(self) -> None:
        self.sync_camera_to_visible_volumes()

    def sync_camera_to_visible_volumes(self) -> None:
        snapshot = self.roi_workspace.renderer.camera_state_for_visible_volumes()
        if snapshot is None:
            snapshot = (
                self.standard_workspace.renderer.camera_state_for_visible_volumes()
            )
        if snapshot is None:
            return
        self.shared_state.camera_snapshot = snapshot
        self.standard_workspace.renderer.apply_camera_state(snapshot)
        self.roi_workspace.renderer.apply_camera_state(snapshot)

    def capture_camera_state(self):
        snapshot = self.active_workspace.renderer.capture_camera_state()
        self.shared_state.camera_snapshot = snapshot
        return snapshot

    def apply_camera_state(self, snapshot) -> None:
        self.shared_state.camera_snapshot = snapshot
        self.standard_workspace.renderer.apply_camera_state(snapshot)
        self.roi_workspace.renderer.apply_camera_state(snapshot)

    def save_screenshot(self, filename: str) -> None:
        self.active_workspace.renderer.save_screenshot(filename)

    def record_rotation_video(self, filename: str, rotation_speed: float) -> None:
        self.active_workspace.renderer.record_rotation_video(filename, rotation_speed)

    def shutdown(self) -> None:
        self.standard_workspace.shutdown()
        self.roi_workspace.shutdown()
