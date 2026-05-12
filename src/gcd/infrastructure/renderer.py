from __future__ import annotations

import imageio
import numpy as np
import vtk
import vtk.util.numpy_support

from src.gcd.presentation.qt.annotation_geometry import has_meaningful_3d_box_drag
from src.utils import timer


def _vtk_direction_matrix(metadata: dict[str, object]) -> np.ndarray | None:
    direction = metadata.get("vtk_direction")
    if direction is None:
        return None
    try:
        axes = np.asarray(direction, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if axes.shape != (3, 3):
        return None
    # Metadata stores one direction vector per VTK axis. VTK's direction matrix
    # maps voxel coordinates to physical space with those vectors as columns.
    return axes.T


def _direction_user_matrix(
    origin: tuple[float, float, float], direction_matrix: np.ndarray | None
):
    if direction_matrix is None:
        return None
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    origin_vector = np.asarray(origin, dtype=np.float32)
    translation = origin_vector - direction_matrix @ origin_vector
    for row in range(3):
        for col in range(3):
            matrix.SetElement(row, col, float(direction_matrix[row, col]))
        matrix.SetElement(row, 3, float(translation[row]))
    return matrix


def _apply_vtk_direction(volume, origin, metadata: dict[str, object]) -> np.ndarray | None:
    direction_matrix = _vtk_direction_matrix(metadata)
    if direction_matrix is None:
        return direction_matrix
    matrix = _direction_user_matrix(origin, direction_matrix)
    if matrix is not None:
        volume.SetUserMatrix(matrix)
    return direction_matrix


def _build_vtk_image_data(data, spacing, origin, metadata: dict[str, object]):
    if hasattr(data, "detach"):
        np_array = np.ascontiguousarray(data.detach().cpu().numpy())
    else:
        np_array = np.ascontiguousarray(np.array(data))

    vtk_array = vtk.util.numpy_support.numpy_to_vtk(
        np_array.ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT
    )
    image_data = vtk.vtkImageData()
    dims = (int(np_array.shape[2]), int(np_array.shape[1]), int(np_array.shape[0]))
    image_data.SetDimensions(*dims)
    vtk_spacing = tuple(metadata.get("vtk_spacing", spacing))
    vtk_origin = tuple(metadata.get("vtk_origin", origin))
    image_data.SetSpacing(*vtk_spacing)
    image_data.SetOrigin(*vtk_origin)
    image_data.GetPointData().SetScalars(vtk_array)
    return np_array, image_data, vtk_spacing, vtk_origin


class Roi3DInteractionController:
    def __init__(self, renderer: "VtkVolumeRenderer") -> None:
        self.renderer = renderer
        self.mode = "off"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.renderer._reset_annotation_interaction_state()

    def handles_left_mouse(self) -> bool:
        return self.mode in {"point", "box"}

    def on_left_button_press(self, obj) -> bool:
        if not self.handles_left_mouse():
            return False
        x, y = self.renderer.interactor.GetEventPosition()
        picked = self.renderer._pick_annotation_actor(x, y)
        if picked is not None:
            kind = picked[0]
            if kind == "box":
                self.renderer._emit_annotation_event(
                    "select_annotation", {"annotation_id": picked[1]}
                )
                if self.mode == "box":
                    world = self.renderer._pick_world(x, y)
                    if world is not None:
                        voxel = self.renderer._clamp_voxel(
                            self.renderer._world_to_voxel(world)
                        )
                        box = self.renderer._box_by_id(picked[1])
                        if box is not None:
                            self.renderer.dragging_box_id = picked[1]
                            self.renderer.dragging_box_anchor = voxel
                            self.renderer.dragging_box_initial_bounds = (
                                tuple(float(v) for v in box["min_corner"]),
                                tuple(float(v) for v in box["max_corner"]),
                            )
                obj.AbortFlagOn()
                return True
            if kind == "point":
                self.renderer._emit_annotation_event(
                    "select_annotation", {"annotation_id": picked[1]}
                )
                obj.AbortFlagOn()
                return True
            if kind == "handle":
                self.renderer._emit_annotation_event(
                    "select_annotation", {"annotation_id": picked[1]}
                )
                if self.mode == "box":
                    self.renderer.dragging_handle = picked
                    self.renderer.dragging_box_id = picked[1]
                    self.renderer.drag_box_corner_index = picked[2]
                obj.AbortFlagOn()
                return True

        world = self.renderer._pick_world(x, y)
        if world is None:
            return False
        voxel = self.renderer._clamp_voxel(self.renderer._world_to_voxel(world))
        if self.mode == "point":
            self.renderer._emit_annotation_event("add_point_3d", {"position": voxel})
            obj.AbortFlagOn()
            return True
        if self.mode == "box":
            self.renderer.box_creation_start = voxel
            self.renderer.box_creation_active = True
            self.renderer.preview_box = (voxel, voxel)
            self.renderer._update_preview_box()
            obj.AbortFlagOn()
            return True
        return False

    def on_mouse_move(self, obj) -> bool:
        if not self.handles_left_mouse():
            return False
        x, y = self.renderer.interactor.GetEventPosition()
        world = self.renderer._pick_world(x, y)
        if world is None:
            return False
        voxel = self.renderer._clamp_voxel(self.renderer._world_to_voxel(world))
        if (
            self.renderer.dragging_handle is not None
            and self.renderer.dragging_box_id is not None
        ):
            self.renderer._emit_annotation_event(
                "resize_box_3d",
                {
                    "annotation_id": self.renderer.dragging_box_id,
                    "corner_index": self.renderer.drag_box_corner_index,
                    "position": voxel,
                },
            )
            obj.AbortFlagOn()
            return True
        if (
            self.renderer.dragging_box_id is not None
            and self.renderer.dragging_box_anchor is not None
            and self.renderer.dragging_box_initial_bounds is not None
        ):
            start = self.renderer.dragging_box_anchor
            delta = tuple(float(voxel[i]) - float(start[i]) for i in range(3))
            self.renderer._emit_annotation_event(
                "move_box_3d",
                {
                    "annotation_id": self.renderer.dragging_box_id,
                    "delta": delta,
                    "initial_min_corner": self.renderer.dragging_box_initial_bounds[0],
                    "initial_max_corner": self.renderer.dragging_box_initial_bounds[1],
                },
            )
            obj.AbortFlagOn()
            return True
        if (
            self.mode == "box"
            and self.renderer.box_creation_active
            and self.renderer.box_creation_start is not None
        ):
            self.renderer.preview_box = (self.renderer.box_creation_start, voxel)
            self.renderer._update_preview_box()
            obj.AbortFlagOn()
            return True
        return False

    def on_left_button_release(self, obj) -> bool:
        if not self.handles_left_mouse():
            return False
        if self.renderer.dragging_handle is not None:
            self.renderer.dragging_handle = None
            self.renderer.dragging_box_id = None
            self.renderer.drag_box_corner_index = None
            self.renderer.dragging_box_anchor = None
            self.renderer.dragging_box_initial_bounds = None
            obj.AbortFlagOn()
            return True
        if self.renderer.dragging_box_anchor is not None:
            self.renderer.dragging_box_id = None
            self.renderer.dragging_box_anchor = None
            self.renderer.dragging_box_initial_bounds = None
            obj.AbortFlagOn()
            return True
        if (
            self.mode != "box"
            or not self.renderer.box_creation_active
            or self.renderer.box_creation_start is None
        ):
            return False
        x, y = self.renderer.interactor.GetEventPosition()
        world = self.renderer._pick_world(x, y)
        if world is None:
            self.renderer._reset_box_creation_state()
            self.renderer.render()
            return False
        voxel = self.renderer._clamp_voxel(self.renderer._world_to_voxel(world))
        if has_meaningful_3d_box_drag(self.renderer.box_creation_start, voxel):
            self.renderer._emit_annotation_event(
                "add_box_3d",
                {"min_corner": self.renderer.box_creation_start, "max_corner": voxel},
            )
        self.renderer._reset_box_creation_state()
        self.renderer.render()
        obj.AbortFlagOn()
        return True


class VtkVolumeRenderer:
    def __init__(self, vtk_widget) -> None:
        self.vtk_widget = vtk_widget
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.camera_interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.annotation_interactor_style = vtk.vtkInteractorStyleUser()
        self.interactor.SetInteractorStyle(self.camera_interactor_style)

        self.volumes = []
        self.rotating = False
        self.timer_id = None
        self.rotation_speed = 0.5
        self.initial_camera = None
        self.observer_tag = None
        self.annotation_mode = "off"
        self.annotation_event_handler = None
        self.annotation_point_size = 8
        self.annotation_points = []
        self.annotation_boxes = []
        self.selected_annotation_id = None
        self.active_roi_box_id = None
        self.annotation_point_actors = {}
        self.annotation_box_actors = {}
        self.annotation_handle_actors = {}
        self.annotation_actor_map = {}
        self.preview_box = None
        self.preview_box_actor = None
        self.dragging_handle = None
        self.dragging_box_id = None
        self.drag_box_corner_index = None
        self.dragging_box_anchor = None
        self.dragging_box_initial_bounds = None
        self.box_creation_start = None
        self.box_creation_active = False
        self.volume_shape = (0, 0, 0)
        self.roi_interaction_controller = Roi3DInteractionController(self)
        self.add_axes_indicator()
        self.annotation_interactor_style.AddObserver(
            "LeftButtonPressEvent", self._on_left_button_press, 1.0
        )
        self.annotation_interactor_style.AddObserver(
            "MouseMoveEvent", self._on_mouse_move, 1.0
        )
        self.annotation_interactor_style.AddObserver(
            "LeftButtonReleaseEvent", self._on_left_button_release, 1.0
        )

    def add_axes_indicator(self) -> None:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)
        axes.SetConeRadius(0.1)
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")

        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)
        self.orientation_widget.SetViewport(0.8, 0.0, 1.0, 0.2)
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()

    def add_volume_data(
        self,
        data,
        spacing,
        color_settings=None,
        opacity_settings=None,
        origin=(0.0, 0.0, 0.0),
        metadata: dict[str, object] | None = None,
    ) -> int:
        metadata = metadata or {}
        np_array, image_data, vtk_spacing, vtk_origin = _build_vtk_image_data(
            data, spacing, origin, metadata
        )

        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.6)
        prop.SetSpecular(0.4)

        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(image_data)

        volume = vtk.vtkVolume()
        volume.SetProperty(prop)
        volume.SetMapper(mapper)
        vtk_direction = _apply_vtk_direction(volume, vtk_origin, metadata)

        cset = color_settings if color_settings is not None else []
        oset = opacity_settings if opacity_settings is not None else []
        self._apply_transfer_functions_to_property(prop, cset, oset)

        port = len(self.volumes)
        self.renderer.AddVolume(volume)

        self.volumes.append(
            {
                "image": image_data,
                "mapper": mapper,
                "volume": volume,
                "prop": prop,
                "color": list(cset),
                "opacity": list(oset),
                "affine": self._render_affine(
                    vtk_origin, vtk_spacing, vtk_direction
                ),
                "inverse_affine": None,
            }
        )
        self.volumes[-1]["inverse_affine"] = self._safe_inverse_affine(
            self.volumes[-1]["affine"]
        )

        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.volume_shape = tuple(int(v) for v in np_array.shape)
        self.renderer.ResetCameraClippingRange()
        self.store_initial_camera()
        self._rebuild_annotation_actors()
        self.render()
        return port

    def clear_volumes(self) -> None:
        self.renderer.RemoveAllViewProps()
        self.add_axes_indicator()
        self.volumes = []
        self.annotation_point_actors = {}
        self.annotation_box_actors = {}
        self.annotation_handle_actors = {}
        self.annotation_actor_map = {}
        self.render()

    def show_volumes(
        self,
        volumes: list[object],
        spacing: list[tuple[float, float, float]],
        metadata: list[dict[str, object] | None] | None = None,
    ) -> None:
        with timer("渲染"):
            self.clear_volumes()
            metadata_items = metadata if metadata is not None else [None] * len(volumes)
            for data, space, meta in zip(volumes, spacing, metadata_items):
                if data is not None and space is not None:
                    self.add_volume_data(data, space, metadata=meta)

    def set_volume_transfer_functions(
        self,
        index,
        color_settings,
        opacity_settings,
        *,
        visible: bool | None = None,
        render: bool = True,
    ) -> None:
        if not (0 <= index < len(self.volumes)):
            return
        volume = self.volumes[index]
        volume["color"] = list(color_settings or [])
        volume["opacity"] = list(opacity_settings or [])
        if visible is not None:
            volume["volume"].SetVisibility(1 if visible else 0)
        self._apply_transfer_functions_to_property(
            volume["prop"], volume["color"], volume["opacity"]
        )
        if render:
            self.render()

    def _apply_transfer_functions_to_property(
        self, prop, color_settings, opacity_settings
    ) -> None:
        pwf = vtk.vtkPiecewiseFunction()
        for value, opacity in opacity_settings:
            pwf.AddPoint(float(value), float(opacity))
        prop.SetScalarOpacity(pwf)

        ctf = vtk.vtkColorTransferFunction()
        for value, red, green, blue in color_settings:
            ctf.AddRGBPoint(float(value), float(red), float(green), float(blue))
        prop.SetColor(ctf)

    def replace_camera(self) -> None:
        snapshot = self.camera_state_for_visible_volumes()
        if snapshot is None:
            return
        self.apply_camera_state(snapshot)
        self.store_initial_camera()

    def camera_state_for_visible_volumes(
        self,
    ) -> dict[str, tuple[float, float, float] | float] | None:
        if not self.volumes:
            return None
        xmin, xmax, ymin, ymax, zmin, zmax = [
            np.inf,
            -np.inf,
            np.inf,
            -np.inf,
            np.inf,
            -np.inf,
        ]
        visible_bounds_found = False
        for volume in self.volumes:
            if hasattr(volume["volume"], "GetVisibility") and not volume["volume"].GetVisibility():
                continue
            bounds = self._volume_bounds(volume)
            visible_bounds_found = True
            xmin = min(xmin, bounds[0])
            xmax = max(xmax, bounds[1])
            ymin = min(ymin, bounds[2])
            ymax = max(ymax, bounds[3])
            zmin = min(zmin, bounds[4])
            zmax = max(zmax, bounds[5])

        if not visible_bounds_found:
            for volume in self.volumes:
                bounds = self._volume_bounds(volume)
                xmin = min(xmin, bounds[0])
                xmax = max(xmax, bounds[1])
                ymin = min(ymin, bounds[2])
                ymax = max(ymax, bounds[3])
                zmin = min(zmin, bounds[4])
                zmax = max(zmax, bounds[5])

        if not np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all():
            return None

        center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5)
        distance = max(xmax - xmin, ymax - ymin, zmax - zmin) * 2.0
        if distance <= 0.0:
            distance = 1.0
        parallel_scale = max(ymax - ymin, xmax - xmin, zmax - zmin) * 0.5
        if parallel_scale <= 0.0:
            parallel_scale = 1.0
        return {
            "position": (center[0], center[1], center[2] + distance),
            "focal_point": center,
            "view_up": (0.0, 1.0, 0.0),
            "parallel_scale": float(parallel_scale),
        }

    @staticmethod
    def _volume_bounds(volume) -> tuple[float, float, float, float, float, float]:
        prop = volume.get("volume")
        if prop is not None and hasattr(prop, "GetBounds"):
            bounds = prop.GetBounds()
            if bounds is not None and VtkVolumeRenderer._bounds_are_valid(bounds):
                return tuple(float(v) for v in bounds)
        return tuple(float(v) for v in volume["image"].GetBounds())

    @staticmethod
    def _bounds_are_valid(bounds) -> bool:
        return (
            bounds is not None
            and np.isfinite(bounds).all()
            and bounds[0] <= bounds[1]
            and bounds[2] <= bounds[3]
            and bounds[4] <= bounds[5]
        )

    def capture_camera_state(
        self,
    ) -> dict[str, tuple[float, float, float] | float] | None:
        if not self.volumes:
            return None
        camera = self.renderer.GetActiveCamera()
        return {
            "position": tuple(float(v) for v in camera.GetPosition()),
            "focal_point": tuple(float(v) for v in camera.GetFocalPoint()),
            "view_up": tuple(float(v) for v in camera.GetViewUp()),
            "parallel_scale": float(camera.GetParallelScale()),
        }

    def apply_camera_state(
        self, snapshot: dict[str, tuple[float, float, float] | float] | None
    ) -> None:
        if not snapshot or not self.volumes:
            return
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*snapshot["position"])
        camera.SetFocalPoint(*snapshot["focal_point"])
        camera.SetViewUp(*snapshot["view_up"])
        if "parallel_scale" in snapshot:
            camera.SetParallelScale(float(snapshot["parallel_scale"]))
        self.renderer.ResetCameraClippingRange()
        self.render()

    def store_initial_camera(self) -> None:
        camera = self.renderer.GetActiveCamera()
        self.initial_camera = vtk.vtkCamera()
        self.initial_camera.DeepCopy(camera)

    def render(self) -> None:
        self.render_window.Render()

    def set_annotation_event_handler(self, handler) -> None:
        self.annotation_event_handler = handler

    def set_annotation_mode(self, mode: str) -> None:
        self.annotation_mode = mode
        self.roi_interaction_controller.set_mode(mode)
        if mode in {"point", "box"}:
            self.interactor.SetInteractorStyle(self.annotation_interactor_style)
        else:
            self.interactor.SetInteractorStyle(self.camera_interactor_style)

    def set_annotations(
        self,
        points,
        boxes_3d,
        selected_annotation_id: str | None,
        active_roi_box_id: str | None,
        point_size: int,
    ) -> None:
        self.annotation_points = list(points)
        self.annotation_boxes = list(boxes_3d)
        self.selected_annotation_id = selected_annotation_id
        self.active_roi_box_id = active_roi_box_id
        self.annotation_point_size = point_size
        self._rebuild_annotation_actors()
        self.render()

    def _remove_preview_box(self) -> None:
        if self.preview_box_actor is not None:
            self.renderer.RemoveActor(self.preview_box_actor)
            self.preview_box_actor = None

    def _reset_box_creation_state(self) -> None:
        self.box_creation_start = None
        self.box_creation_active = False
        self.preview_box = None
        self._remove_preview_box()

    def _reset_annotation_interaction_state(self) -> None:
        self.dragging_handle = None
        self.dragging_box_id = None
        self.drag_box_corner_index = None
        self.dragging_box_anchor = None
        self.dragging_box_initial_bounds = None
        self._reset_box_creation_state()

    def _rebuild_annotation_actors(self) -> None:
        for actor in self.annotation_point_actors.values():
            self.renderer.RemoveActor(actor)
        for actor in self.annotation_box_actors.values():
            self.renderer.RemoveActor(actor)
        for actor in self.annotation_handle_actors.values():
            self.renderer.RemoveActor(actor)
        self.annotation_point_actors = {}
        self.annotation_box_actors = {}
        self.annotation_handle_actors = {}
        self.annotation_actor_map = {}
        self._remove_preview_box()

        if not self.volumes:
            return

        for item in self.annotation_points:
            actor = self._build_point_actor(item.position, item.size)
            self.annotation_point_actors[item.id] = actor
            self.annotation_actor_map[actor] = ("point", item.id)
            self.renderer.AddActor(actor)

        for item in self.annotation_boxes:
            actor = self._build_box_actor(
                item.min_corner,
                item.max_corner,
                highlight=(item.id == self.active_roi_box_id),
                selected=(item.id == self.selected_annotation_id),
            )
            self.annotation_box_actors[item.id] = actor
            self.annotation_actor_map[actor] = ("box", item.id)
            self.renderer.AddActor(actor)
            if item.id == self.selected_annotation_id:
                self._add_box_handles(item)

    def _voxel_to_world(self, voxel) -> tuple[float, float, float]:
        if not self.volumes:
            return tuple(float(v) for v in voxel)
        affine = self.volumes[0]["affine"]
        ijk = np.array([float(voxel[0]), float(voxel[1]), float(voxel[2]), 1.0])
        world = affine @ ijk
        return tuple(float(world[i]) for i in range(3))

    def _world_to_voxel(self, world) -> tuple[float, float, float]:
        if not self.volumes:
            return tuple(float(v) for v in world)
        inverse_affine = self.volumes[0]["inverse_affine"]
        point = np.array([float(world[0]), float(world[1]), float(world[2]), 1.0])
        voxel = inverse_affine @ point
        return tuple(float(voxel[i]) for i in range(3))

    @staticmethod
    def _render_affine(origin, spacing, direction=None) -> np.ndarray:
        affine = np.eye(4, dtype=np.float32)
        affine[:3, 3] = np.array(origin, dtype=np.float32)
        # Workspace voxel order is (axis0, axis1, axis2), while VTK renders the
        # same numpy buffer as (x=axis2, y=axis1, z=axis0).
        permutation = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        spacing_matrix = np.diag(
            [float(spacing[0]), float(spacing[1]), float(spacing[2])]
        ).astype(np.float32)
        direction_matrix = (
            np.asarray(direction, dtype=np.float32)
            if direction is not None
            else np.eye(3, dtype=np.float32)
        )
        affine[:3, :3] = direction_matrix @ spacing_matrix @ permutation
        return affine

    @staticmethod
    def _safe_inverse_affine(affine: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(affine)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(affine)

    def _build_point_actor(self, voxel_position, size: int):
        world = self._voxel_to_world(voxel_position)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*world)
        scale = max(self.volumes[0]["image"].GetSpacing()) if self.volumes else 1.0
        sphere.SetRadius(max(0.5, size / 8.0) * scale)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.95, 0.35, 0.2)
        return actor

    def _build_box_actor(
        self, min_corner, max_corner, *, highlight: bool, selected: bool
    ):
        source = vtk.vtkOutlineSource()
        world_min = self._voxel_to_world(min_corner)
        world_max = self._voxel_to_world(max_corner)
        source.SetBounds(
            world_min[0],
            world_max[0],
            world_min[1],
            world_max[1],
            world_min[2],
            world_max[2],
        )
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if highlight:
            actor.GetProperty().SetColor(0.95, 0.78, 0.2)
            actor.GetProperty().SetLineWidth(3.5)
        elif selected:
            actor.GetProperty().SetColor(0.2, 0.85, 0.95)
            actor.GetProperty().SetLineWidth(3.0)
        else:
            actor.GetProperty().SetColor(0.3, 0.95, 0.55)
            actor.GetProperty().SetLineWidth(2.0)
        return actor

    def _add_box_handles(self, box) -> None:
        min_corner = box.min_corner
        max_corner = box.max_corner
        corners = [
            (min_corner[0], min_corner[1], min_corner[2]),
            (max_corner[0], min_corner[1], min_corner[2]),
            (min_corner[0], max_corner[1], min_corner[2]),
            (max_corner[0], max_corner[1], min_corner[2]),
            (min_corner[0], min_corner[1], max_corner[2]),
            (max_corner[0], min_corner[1], max_corner[2]),
            (min_corner[0], max_corner[1], max_corner[2]),
            (max_corner[0], max_corner[1], max_corner[2]),
        ]
        scale = max(self.volumes[0]["image"].GetSpacing()) if self.volumes else 1.0
        for index, corner in enumerate(corners):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*self._voxel_to_world(corner))
            sphere.SetRadius(1.5 * scale)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.95, 0.95, 0.2)
            self.annotation_handle_actors[(box.id, index)] = actor
            self.annotation_actor_map[actor] = ("handle", box.id, index)
            self.renderer.AddActor(actor)

    def _box_by_id(self, annotation_id: str):
        for item in self.annotation_boxes:
            if item.id == annotation_id:
                return item
        return None

    def _pick_world(self, x: int, y: int):
        if not self.volumes:
            return None
        volume_picker = vtk.vtkVolumePicker()
        volume_picker.SetTolerance(0.0005)
        if volume_picker.Pick(x, y, 0, self.renderer):
            picked_volume = volume_picker.GetVolume()
            if picked_volume is None:
                picked_volume = volume_picker.GetProp3D()
            volume_index = self._volume_index_for_prop(picked_volume)
            if volume_index is None:
                return None
            position = tuple(float(v) for v in volume_picker.GetPickPosition())
            if all(np.isfinite(value) for value in position):
                return position
        return None

    def _volume_index_for_prop(self, prop) -> int | None:
        if prop is None:
            return None
        for index, item in enumerate(self.volumes):
            volume = item["volume"]
            if volume is prop or volume == prop:
                return index
            try:
                if volume.GetAddressAsString("") == prop.GetAddressAsString(""):
                    return index
            except AttributeError:
                continue
        return None

    def _pick_annotation_actor(self, x: int, y: int):
        picker = vtk.vtkPropPicker()
        if picker.Pick(x, y, 0, self.renderer):
            actor = picker.GetActor()
            return self.annotation_actor_map.get(actor)
        return None

    def _clamp_voxel(self, voxel) -> tuple[float, float, float]:
        if not any(self.volume_shape):
            return tuple(float(v) for v in voxel)
        return tuple(
            max(0.0, min(float(voxel[i]), float(self.volume_shape[i] - 1)))
            for i in range(3)
        )

    def _emit_annotation_event(self, event_type: str, payload: dict) -> None:
        if self.annotation_event_handler is not None:
            self.annotation_event_handler(event_type, payload)

    def _on_left_button_press(self, obj, event) -> None:
        self.roi_interaction_controller.on_left_button_press(obj)

    def _on_mouse_move(self, obj, event) -> None:
        self.roi_interaction_controller.on_mouse_move(obj)

    def _update_preview_box(self) -> None:
        if self.preview_box is None:
            self._remove_preview_box()
            return
        self._remove_preview_box()
        start, end = self.preview_box
        self.preview_box_actor = self._build_box_actor(
            tuple(min(start[i], end[i]) for i in range(3)),
            tuple(max(start[i], end[i]) for i in range(3)),
            highlight=False,
            selected=True,
        )
        self.preview_box_actor.GetProperty().SetOpacity(0.7)
        self.renderer.AddActor(self.preview_box_actor)
        self.render()

    def _on_left_button_release(self, obj, event) -> None:
        self.roi_interaction_controller.on_left_button_release(obj)

    def set_rotation_speed(self, speed: float) -> None:
        self.rotation_speed = speed

    def start_rotation(self) -> None:
        if self.rotating:
            self.stop_rotation()
        if self.rotation_speed <= 0:
            return

        base_speed = 30.0
        timer_interval = 30

        def rotate_callback(_obj, _event) -> None:
            degrees_per_second = base_speed * self.rotation_speed
            angle_step = degrees_per_second * (timer_interval / 1000.0)
            camera = self.renderer.GetActiveCamera()
            camera.Azimuth(angle_step)
            self.renderer.ResetCameraClippingRange()
            self.render_window.Render()

        self.observer_tag = self.interactor.AddObserver("TimerEvent", rotate_callback)
        self.timer_id = self.interactor.CreateRepeatingTimer(timer_interval)
        self.rotating = True

    def stop_rotation(self) -> None:
        if self.rotating and self.interactor:
            self.rotating = False
            if self.timer_id is not None:
                self.interactor.DestroyTimer(self.timer_id)
                self.timer_id = None
            if self.observer_tag is not None:
                self.interactor.RemoveObserver(self.observer_tag)
                self.observer_tag = None

    def shutdown(self) -> None:
        self.stop_rotation()
        try:
            if (
                hasattr(self, "orientation_widget")
                and self.orientation_widget is not None
            ):
                self.orientation_widget.SetEnabled(0)
                self.orientation_widget.SetInteractor(None)
        except Exception:
            pass
        try:
            self.renderer.RemoveAllViewProps()
        except Exception:
            pass
        try:
            if self.render_window is not None:
                self.render_window.Finalize()
        except Exception:
            pass
        try:
            if self.vtk_widget is not None and hasattr(self.vtk_widget, "Finalize"):
                self.vtk_widget.Finalize()
        except Exception:
            pass

    def save_screenshot(self, filename: str) -> None:
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(f"{filename}.png")
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def record_rotation_video(self, filename: str, rotation_speed: float) -> None:
        if len(self.volumes) == 0 or rotation_speed <= 0:
            return
        total_rotation = 360
        fps = 30
        base_speed = 60.0
        degrees_per_second = base_speed * rotation_speed
        recording_time = total_rotation / degrees_per_second
        total_frames = max(int(recording_time * fps), 30)
        angle_step = total_rotation / total_frames

        if not filename.endswith(".mp4"):
            filename += ".mp4"
        writer = imageio.get_writer(filename, fps=fps, codec="libx264", quality=8)
        self.render_window.SetSize(1328, 960)
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetScale(1)

        camera = self.renderer.GetActiveCamera()
        initial_position = camera.GetPosition()

        for _frame in range(total_frames):
            camera.Azimuth(angle_step)
            self.render_window.Render()

            window_to_image_filter.Modified()
            window_to_image_filter.Update()
            vtk_image = window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            numpy_array = np.flipud(
                vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, 3)
            )
            writer.append_data(numpy_array)

        writer.close()
        camera.SetPosition(*initial_position)
        self.render()


class StandardMultiVolumeRenderer(VtkVolumeRenderer):
    def __init__(self, vtk_widget) -> None:
        super().__init__(vtk_widget)
        self._reset_multi_volume_backend()

    def _reset_multi_volume_backend(self) -> None:
        self.multi_mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.multi_volume = vtk.vtkMultiVolume()
        self.multi_volume.SetMapper(self.multi_mapper)
        self.multi_volume_added = False
        self.multi_volume_dummy_port = None

    def _remove_multi_volume_from_renderer(self) -> None:
        if not self.multi_volume_added:
            return
        self.renderer.RemoveVolume(self.multi_volume)
        self.multi_volume_added = False

    def show_volumes(
        self,
        volumes: list[object],
        spacing: list[tuple[float, float, float]],
        metadata: list[dict[str, object] | None] | None = None,
    ) -> None:
        valid_items = [
            (data, space, meta)
            for data, space, meta in zip(
                volumes,
                spacing,
                metadata if metadata is not None else [None] * len(volumes),
            )
            if data is not None and space is not None
        ]
        with timer("渲染"):
            self.clear_volumes()
            for data, space, meta in valid_items:
                self.add_volume_data(data, space, metadata=meta)
            self._ensure_single_volume_multi_input()

    def add_volume_data(
        self,
        data,
        spacing,
        color_settings=None,
        opacity_settings=None,
        origin=(0.0, 0.0, 0.0),
        metadata: dict[str, object] | None = None,
    ) -> int:
        metadata = metadata or {}
        np_array, image_data, vtk_spacing, vtk_origin = _build_vtk_image_data(
            data, spacing, origin, metadata
        )

        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.6)
        prop.SetSpecular(0.4)

        child_volume = vtk.vtkVolume()
        child_volume.SetProperty(prop)
        vtk_direction = _apply_vtk_direction(child_volume, vtk_origin, metadata)

        cset = color_settings if color_settings is not None else []
        oset = opacity_settings if opacity_settings is not None else []
        self._apply_transfer_functions_to_property(prop, cset, oset)

        port = len(self.volumes)
        self.multi_mapper.SetInputDataObject(port, image_data)
        self.multi_volume.SetVolume(child_volume, port)
        child_volume.Modified()
        prop.Modified()
        self.multi_mapper.Modified()
        self.multi_volume.Modified()
        if not self.multi_volume_added:
            self.renderer.AddVolume(self.multi_volume)
            self.multi_volume_added = True

        self.volumes.append(
            {
                "image": image_data,
                "mapper": self.multi_mapper,
                "volume": child_volume,
                "prop": prop,
                "color": list(cset),
                "opacity": list(oset),
                "affine": self._render_affine(
                    vtk_origin, vtk_spacing, vtk_direction
                ),
                "inverse_affine": None,
                "volume_id": str(metadata.get("volume_id", port)),
                "visible": True,
                "render_port": port,
            }
        )
        self.volumes[-1]["inverse_affine"] = self._safe_inverse_affine(
            self.volumes[-1]["affine"]
        )

        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.volume_shape = tuple(int(v) for v in np_array.shape)
        self.renderer.ResetCameraClippingRange()
        self.store_initial_camera()
        self.multi_volume.Modified()
        self.render()
        return port

    def clear_volumes(self) -> None:
        self.renderer.RemoveAllViewProps()
        self.add_axes_indicator()
        self._reset_multi_volume_backend()
        self.volumes = []
        self.annotation_point_actors = {}
        self.annotation_box_actors = {}
        self.annotation_handle_actors = {}
        self.annotation_actor_map = {}
        self.render()

    def _visible_multi_volume_items(self):
        return [
            (index, volume)
            for index, volume in enumerate(self.volumes)
            if bool(volume.get("visible", True))
        ]

    def _add_single_volume_dummy_input(self, source_volume, port: int) -> None:
        if self.multi_volume_dummy_port is not None:
            return

        dummy_prop = vtk.vtkVolumeProperty()
        dummy_prop.SetInterpolationTypeToLinear()
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(0.0, 0.0)
        opacity.AddPoint(1.0, 0.0)
        dummy_prop.SetScalarOpacity(opacity)
        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color.AddRGBPoint(1.0, 0.0, 0.0, 0.0)
        dummy_prop.SetColor(color)

        dummy_volume = vtk.vtkVolume()
        dummy_volume.SetProperty(dummy_prop)
        source_matrix = source_volume["volume"].GetUserMatrix()
        if source_matrix is not None:
            matrix = vtk.vtkMatrix4x4()
            matrix.DeepCopy(source_matrix)
            dummy_volume.SetUserMatrix(matrix)

        self.multi_mapper.SetInputDataObject(port, source_volume["image"])
        self.multi_volume.SetVolume(dummy_volume, port)
        dummy_volume.Modified()
        dummy_prop.Modified()
        self.multi_mapper.Modified()
        self.multi_volume.Modified()
        self.multi_volume_dummy_port = port

    def _ensure_single_volume_multi_input(self) -> None:
        visible_items = self._visible_multi_volume_items()
        if len(visible_items) != 1:
            return
        self._add_single_volume_dummy_input(visible_items[0][1], len(visible_items))

    def _rebuild_visible_multi_volume(self) -> None:
        self._remove_multi_volume_from_renderer()
        self._reset_multi_volume_backend()
        visible_items = self._visible_multi_volume_items()

        for port, (_original_index, volume) in enumerate(visible_items):
            volume["mapper"] = self.multi_mapper
            volume["render_port"] = port
            volume["volume"].SetVisibility(1)
            volume["volume"].Modified()
            volume["prop"].Modified()
            self.multi_mapper.SetInputDataObject(port, volume["image"])
            self.multi_volume.SetVolume(volume["volume"], port)

        visible_ids = {id(volume) for _index, volume in visible_items}
        for volume in self.volumes:
            if id(volume) not in visible_ids:
                volume["render_port"] = None

        if len(visible_items) == 1:
            self._add_single_volume_dummy_input(visible_items[0][1], len(visible_items))

        if visible_items:
            self.renderer.AddVolume(self.multi_volume)
            self.multi_volume_added = True

        self.multi_mapper.Modified()
        self.multi_volume.Modified()
        self.renderer.ResetCameraClippingRange()

    def set_volume_transfer_functions(
        self,
        index,
        color_settings,
        opacity_settings,
        *,
        visible: bool | None = None,
        render: bool = True,
    ) -> None:
        if not (0 <= index < len(self.volumes)):
            return
        volume = self.volumes[index]
        volume["color"] = list(color_settings or [])
        volume["opacity"] = list(opacity_settings or [])
        if visible is not None:
            volume["visible"] = bool(visible)
            volume["volume"].SetVisibility(1 if visible else 0)
            volume["volume"].Modified()
        self._apply_transfer_functions_to_property(
            volume["prop"], volume["color"], volume["opacity"]
        )
        volume["prop"].Modified()
        if visible is not None:
            self._rebuild_visible_multi_volume()
        else:
            render_port = volume.get("render_port")
            if render_port is not None:
                self.multi_volume.SetVolume(volume["volume"], int(render_port))
                self.multi_volume.Modified()
                self.multi_mapper.Modified()
        self._rebuild_annotation_actors()
        if render:
            self.render()
