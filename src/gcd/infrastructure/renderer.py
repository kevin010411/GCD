from __future__ import annotations

import imageio
import numpy as np
import vtk
import vtk.util.numpy_support

from src.utils import timer


class VtkVolumeRenderer:
    def __init__(self, vtk_widget) -> None:
        self.vtk_widget = vtk_widget
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

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
        self.add_axes_indicator()

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
    ) -> int:
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
        image_data.SetSpacing(*spacing)
        image_data.SetOrigin(*origin)
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
            }
        )

        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.ResetCameraClippingRange()
        self.store_initial_camera()
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
        self.render()

    def show_volumes(self, volumes: list[object], spacing: list[tuple[float, float, float]]) -> None:
        with timer("渲染"):
            self.clear_volumes()
            for data, space in zip(volumes, spacing):
                if data is not None and space is not None:
                    self.add_volume_data(data, space)

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

    def store_initial_camera(self) -> None:
        camera = self.renderer.GetActiveCamera()
        self.initial_camera = vtk.vtkCamera()
        self.initial_camera.DeepCopy(camera)

    def render(self) -> None:
        self.render_window.Render()

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
