from __future__ import annotations

import imageio
import os
import numpy as np
import vtk
import vtk.util.numpy_support

from src.gcd.presentation.qt.annotation_geometry import has_meaningful_3d_box_drag
from src.utils import timer


class Roi3DInteractionController:
    def __init__(self, renderer: "VtkVolumeRenderer") -> None:
        self.renderer = renderer
        self.mode = "off"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.renderer._debug(f"controller mode -> {mode}")
        self.renderer._reset_annotation_interaction_state()

    def handles_left_mouse(self) -> bool:
        return self.mode in {"point", "box"}

    def on_left_button_press(self, obj) -> bool:
        if not self.handles_left_mouse():
            return False
        x, y = self.renderer.interactor.GetEventPosition()
        self.renderer._debug(f"left press mode={self.mode} screen=({x}, {y})")
        picked = self.renderer._pick_annotation_actor(x, y)
        if picked is not None:
            self.renderer._debug(f"annotation actor picked: {picked}")
            kind = picked[0]
            if kind == "box":
                self.renderer._emit_annotation_event(
                    "select_annotation", {"annotation_id": picked[1]}
                )
                if self.mode == "box":
                    world = self.renderer._pick_world(x, y)
                    if world is not None:
                        voxel = self.renderer._clamp_voxel(self.renderer._world_to_voxel(world))
                        box = self.renderer._box_by_id(picked[1])
                        if box is not None:
                            self.renderer.dragging_box_id = picked[1]
                            self.renderer.dragging_box_anchor = voxel
                            self.renderer.dragging_box_initial_bounds = (
                                tuple(float(v) for v in box["min_corner"]),
                                tuple(float(v) for v in box["max_corner"]),
                            )
                            self.renderer._debug(
                                f"start moving box {picked[1]} from anchor voxel {voxel}"
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
            self.renderer._debug("left press world pick failed")
            return False
        self.renderer._debug(f"left press world picked: {world}")
        voxel = self.renderer._clamp_voxel(self.renderer._world_to_voxel(world))
        self.renderer._debug(f"left press voxel: {voxel}")
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
            self.renderer._debug(f"mouse move mode={self.mode} screen=({x}, {y}) world pick failed")
            return False
        voxel = self.renderer._clamp_voxel(self.renderer._world_to_voxel(world))
        self.renderer._debug(f"mouse move mode={self.mode} screen=({x}, {y}) voxel={voxel}")
        if self.renderer.dragging_handle is not None and self.renderer.dragging_box_id is not None:
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
        if self.mode == "box" and self.renderer.box_creation_active and self.renderer.box_creation_start is not None:
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
        if self.mode != "box" or not self.renderer.box_creation_active or self.renderer.box_creation_start is None:
            return False
        x, y = self.renderer.interactor.GetEventPosition()
        self.renderer._debug(f"left release mode={self.mode} screen=({x}, {y})")
        world = self.renderer._pick_world(x, y)
        if world is None:
            self.renderer._debug("left release world pick failed")
            self.renderer._reset_box_creation_state()
            self.renderer.render()
            return False
        voxel = self.renderer._clamp_voxel(self.renderer._world_to_voxel(world))
        self.renderer._debug(f"left release voxel: {voxel}")
        if has_meaningful_3d_box_drag(self.renderer.box_creation_start, voxel):
            self.renderer._emit_annotation_event(
                "add_box_3d",
                {"min_corner": self.renderer.box_creation_start, "max_corner": voxel},
            )
        else:
            self.renderer._debug(
                f"box drag too small: start={self.renderer.box_creation_start}, end={voxel}"
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

        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.multi_volume = vtk.vtkMultiVolume()
        self.multi_volume.SetMapper(self.mapper)
        self.renderer.AddViewProp(self.multi_volume)

        self.volumes = []
        self.rotating = False
        self.timer_id = None
        self.rotation_speed = 0.5
        self.initial_camera = None
        self.observer_tag = None
        self.annotation_mode = "off"
        self.debug_enabled = os.environ.get("GCD_ROI_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
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

    def _debug(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[ROI3D] {message}")

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

        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.6)
        prop.SetSpecular(0.4)

        volume = vtk.vtkVolume()
        volume.SetProperty(prop)
        volume.SetMapper(self.mapper)

        cset = color_settings if color_settings is not None else []
        oset = opacity_settings if opacity_settings is not None else []
        self._apply_transfer_functions_to_property(prop, cset, oset)

        port = len(self.volumes)
        self.mapper.SetInputDataObject(port, image_data)
        try:
            self.multi_volume.SetVolume(volume, port)
        except AttributeError:
            self.multi_volume.AddVolume(volume)

        self.volumes.append(
            {
                "image": image_data,
                "volume": volume,
                "prop": prop,
                "color": list(cset),
                "opacity": list(oset),
                "affine": self._render_affine(vtk_origin, vtk_spacing),
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
        for index, _volume in enumerate(self.volumes):
            try:
                self.multi_volume.RemoveVolume(index)
            except AttributeError:
                pass

        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.multi_volume = vtk.vtkMultiVolume()
        self.multi_volume.SetMapper(self.mapper)
        self.renderer.RemoveAllViewProps()
        self.renderer.AddViewProp(self.multi_volume)
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
            metadata_items = (
                metadata if metadata is not None else [None] * len(volumes)
            )
            for data, space, meta in zip(volumes, spacing, metadata_items):
                if data is not None and space is not None:
                    self.add_volume_data(data, space, metadata=meta)

    def apply_transfer_function(
        self,
        color_points: list[tuple[float, float, float, float]],
        opacity_points: list[tuple[float, float]],
    ) -> None:
        for index in range(len(self.volumes)):
            self.set_volume_transfer_functions(index, color_points, opacity_points)
        self.render()

    def set_volume_transfer_functions(self, index, color_settings, opacity_settings) -> None:
        if not (0 <= index < len(self.volumes)):
            return
        volume = self.volumes[index]
        volume["color"] = list(color_settings or [])
        volume["opacity"] = list(opacity_settings or [])
        self._apply_transfer_functions_to_property(
            volume["prop"], volume["color"], volume["opacity"]
        )

    def _apply_transfer_functions_to_property(self, prop, color_settings, opacity_settings) -> None:
        pwf = vtk.vtkPiecewiseFunction()
        for value, opacity in opacity_settings:
            pwf.AddPoint(float(value), float(opacity))
        prop.SetScalarOpacity(pwf)

        ctf = vtk.vtkColorTransferFunction()
        for value, red, green, blue in color_settings:
            ctf.AddRGBPoint(float(value), float(red), float(green), float(blue))
        prop.SetColor(ctf)

    def replace_camera(self) -> None:
        if not self.volumes:
            return

        xmin, xmax, ymin, ymax, zmin, zmax = [
            np.inf,
            -np.inf,
            np.inf,
            -np.inf,
            np.inf,
            -np.inf,
        ]
        for volume in self.volumes:
            bounds = volume["image"].GetBounds()
            xmin = min(xmin, bounds[0])
            xmax = max(xmax, bounds[1])
            ymin = min(ymin, bounds[2])
            ymax = max(ymax, bounds[3])
            zmin = min(zmin, bounds[4])
            zmax = max(zmax, bounds[5])

        center = [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5]
        distance = max(xmax - xmin, ymax - ymin, zmax - zmin) * 2.0
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(center[0], center[1], center[2] + distance)
        camera.SetFocalPoint(*center)
        camera.SetViewUp(0, 1, 0)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def capture_camera_state(self) -> dict[str, tuple[float, float, float] | float] | None:
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
        self._debug(f"renderer annotation mode -> {mode}")
        self.roi_interaction_controller.set_mode(mode)
        if mode in {"point", "box"}:
            self._debug("switching to annotation interactor style")
            self.interactor.SetInteractorStyle(self.annotation_interactor_style)
        else:
            self._debug("switching to camera interactor style")
            self.interactor.SetInteractorStyle(self.camera_interactor_style)

    def set_annotations(
        self, points, boxes_3d, selected_annotation_id: str | None, active_roi_box_id: str | None, point_size: int
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
    def _render_affine(origin, spacing) -> np.ndarray:
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
        affine[:3, :3] = permutation @ np.diag(
            [float(spacing[0]), float(spacing[1]), float(spacing[2])]
        )
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

    def _build_box_actor(self, min_corner, max_corner, *, highlight: bool, selected: bool):
        source = vtk.vtkOutlineSource()
        world_min = self._voxel_to_world(min_corner)
        world_max = self._voxel_to_world(max_corner)
        source.SetBounds(
            world_min[0], world_max[0], world_min[1], world_max[1], world_min[2], world_max[2]
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
            self._debug("pick world skipped: no volumes loaded")
            return None
        volume_picker = vtk.vtkVolumePicker()
        volume_picker.SetTolerance(0.0005)
        if volume_picker.Pick(x, y, 0, self.renderer):
            position = tuple(float(v) for v in volume_picker.GetPickPosition())
            if any(np.isfinite(value) for value in position):
                self._debug(f"volume picker hit at {position}")
                return position
        self._debug("volume picker missed")

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        if picker.Pick(x, y, 0, self.renderer):
            position = tuple(float(v) for v in picker.GetPickPosition())
            if any(np.isfinite(value) for value in position):
                self._debug(f"cell picker hit at {position}")
                return position
        self._debug("cell picker missed")
        bounds_hit = self._project_display_to_volume_bounds(x, y)
        if bounds_hit is not None:
            self._debug(f"volume bounds fallback hit at {bounds_hit}")
            return bounds_hit
        fallback = self._project_display_to_focal_plane(x, y)
        if fallback is not None:
            self._debug(f"focal plane fallback hit at {fallback}")
            return fallback
        self._debug("focal plane fallback missed")
        return None

    def _project_display_to_volume_bounds(self, x: int, y: int) -> tuple[float, float, float] | None:
        if not self.volumes:
            return None
        bounds = self.volumes[0]["image"].GetBounds()
        near_world = self._display_to_world(x, y, 0.0)
        far_world = self._display_to_world(x, y, 1.0)
        if near_world is None or far_world is None:
            return None
        origin = np.array(near_world, dtype=np.float64)
        direction = np.array(far_world, dtype=np.float64) - origin
        t_min = 0.0
        t_max = 1.0
        for axis in range(3):
            axis_min = float(bounds[axis * 2])
            axis_max = float(bounds[axis * 2 + 1])
            if abs(direction[axis]) < 1e-8:
                if origin[axis] < axis_min or origin[axis] > axis_max:
                    return None
                continue
            inv = 1.0 / direction[axis]
            t1 = (axis_min - origin[axis]) * inv
            t2 = (axis_max - origin[axis]) * inv
            low = min(t1, t2)
            high = max(t1, t2)
            t_min = max(t_min, low)
            t_max = min(t_max, high)
            if t_min > t_max:
                return None
        hit = origin + direction * t_min
        world = tuple(float(value) for value in hit)
        if not all(np.isfinite(value) for value in world):
            return None
        return world

    def _display_to_world(self, x: int, y: int, z: float) -> tuple[float, float, float] | None:
        self.renderer.SetDisplayPoint(float(x), float(y), float(z))
        self.renderer.DisplayToWorld()
        world_point = self.renderer.GetWorldPoint()
        if not world_point or abs(float(world_point[3])) < 1e-8:
            return None
        world = tuple(float(world_point[i] / world_point[3]) for i in range(3))
        if not all(np.isfinite(value) for value in world):
            return None
        return world

    def _project_display_to_focal_plane(self, x: int, y: int) -> tuple[float, float, float] | None:
        if not self.volumes:
            return None
        camera = self.renderer.GetActiveCamera()
        focal_point = camera.GetFocalPoint()
        self.renderer.SetWorldPoint(
            float(focal_point[0]),
            float(focal_point[1]),
            float(focal_point[2]),
            1.0,
        )
        self.renderer.WorldToDisplay()
        display_point = self.renderer.GetDisplayPoint()
        display_z = float(display_point[2])
        self.renderer.SetDisplayPoint(float(x), float(y), display_z)
        self.renderer.DisplayToWorld()
        world_point = self.renderer.GetWorldPoint()
        if not world_point or abs(float(world_point[3])) < 1e-8:
            return None
        world = tuple(float(world_point[i] / world_point[3]) for i in range(3))
        if not all(np.isfinite(value) for value in world):
            return None
        return world

    def _pick_annotation_actor(self, x: int, y: int):
        picker = vtk.vtkPropPicker()
        if picker.Pick(x, y, 0, self.renderer):
            actor = picker.GetActor()
            picked = self.annotation_actor_map.get(actor)
            self._debug(f"prop picker hit actor -> {picked}")
            return picked
        self._debug("prop picker missed")
        return None

    def _clamp_voxel(self, voxel) -> tuple[float, float, float]:
        if not any(self.volume_shape):
            return tuple(float(v) for v in voxel)
        return tuple(
            max(0.0, min(float(voxel[i]), float(self.volume_shape[i] - 1))) for i in range(3)
        )

    def _emit_annotation_event(self, event_type: str, payload: dict) -> None:
        self._debug(f"emit event {event_type}: {payload}")
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
