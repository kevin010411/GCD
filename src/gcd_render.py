import sys
import vtk, vtk.util.numpy_support
import numpy as np
import imageio
from .utils import timer


class VTKRenderer:
    def __init__(self, vtk_widget, main_window=None):
        self.vtk_widget = vtk_widget
        self.main_window = main_window
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # === Multi-volume members ===
        self.mapper = vtk.vtkGPUVolumeRayCastMapper()  # 單一 mapper，支援多輸入
        self.multi_volume = vtk.vtkMultiVolume()  # 收納多個 vtkVolume 的容器
        self.multi_volume.SetMapper(self.mapper)
        self.renderer.AddViewProp(self.multi_volume)

        # 以列表管理每個 volume 的資源
        # 元素結構: {"image": vtkImageData, "volume": vtkVolume, "prop": vtkVolumeProperty,
        #           "color": [(v,r,g,b),...], "opacity":[(v,a),...]}
        self.volumes = []

        # Initialize as an empty list, waiting for external setup
        self.color_settings = []
        self.opacity_settings = []

        self.rotating = False
        self.timer_id = None
        self.rotation_speed = 0.5  # Default rotation speed

        self.initial_camera = None  # Store initial camera state
        # self.setup_default_camera()
        self.frame_count = 0  # Frame count for saving screenshots
        self.observer_tag = None  # New: Store the observer tag

        # Add axes indicator
        self.add_axes_indicator()

    def add_axes_indicator(self):
        """Add a 3D axes indicator to the renderer"""
        # Create axes actor
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)  # Set the total length of the axes
        axes.SetShaftTypeToCylinder()  # Set the shaft type to cylinder
        axes.SetCylinderRadius(0.02)  # Set the cylinder radius
        axes.SetConeRadius(0.1)  # Set the cone radius

        # Set the labels
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")

        # Create an orientation marker widget
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)

        # Set the orientation marker to the lower right corner
        self.orientation_widget.SetViewport(
            0.8, 0.0, 1.0, 0.2
        )  # (xmin, ymin, xmax, ymax)

        # Set the background color of the orientation marker to match the renderer
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()  # Make the orientation marker non-interactive

    def setup_default_camera(self):
        """Setup a default camera state with Y-axis upward"""
        self.initial_camera = vtk.vtkCamera()
        self.initial_camera.SetPosition(0, 0, 500)
        self.initial_camera.SetFocalPoint(0, 0, 0)
        self.initial_camera.SetViewUp(0, 1, 0)
        self.initial_camera.SetViewAngle(30.0)

    # ---------- Core: add / clear volumes ----------
    def add_volume_data(
        self,
        data,
        spacing,
        color_settings=None,
        opacity_settings=None,
        origin=(0.0, 0.0, 0.0),
    ):
        """
        新增一個 volume 到場景中。
        data: torch.Tensor 或 numpy.ndarray，形狀 (D, H, W)
        spacing: (sx, sy, sz)
        color_settings: [(value, r, g, b), ...]
        opacity_settings: [(value, a), ...]
        origin: (ox, oy, oz)
        """
        # to numpy float32 contiguous
        if hasattr(data, "detach"):
            np_array = np.ascontiguousarray(data.detach().cpu().numpy())
        else:
            np_array = np.ascontiguousarray(np.array(data))

        vtk_array = vtk.util.numpy_support.numpy_to_vtk(
            np_array.ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT
        )

        image_data = vtk.vtkImageData()
        # VTK 的維度順序是 (X, Y, Z) = (W, H, D)
        dims = (int(np_array.shape[2]), int(np_array.shape[1]), int(np_array.shape[0]))
        image_data.SetDimensions(*dims)
        image_data.SetSpacing(*spacing)
        image_data.SetOrigin(*origin)
        image_data.GetPointData().SetScalars(vtk_array)

        # 建立針對此 volume 的 property 與 vtkVolume
        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.6)
        prop.SetSpecular(0.4)

        vol = vtk.vtkVolume()
        vol.SetProperty(prop)
        vol.SetMapper(self.mapper)  # 所有 volume 共用同一個 mapper

        # 設定此 volume 的顏色/不透明度
        cset = color_settings if color_settings is not None else []
        oset = opacity_settings if opacity_settings is not None else []
        self._apply_transfer_functions_to_property(prop, cset, oset)

        # 將此影像接到 mapper 的新輸入埠
        port = len(self.volumes)
        self.mapper.SetInputDataObject(port, image_data)

        # 將 volume 放進 multi_volume
        # 若有 SetVolume(vol,index) 則用它；若 VTK 版本僅提供 AddVolume 也可改用 AddVolume
        try:
            self.multi_volume.SetVolume(vol, port)
        except AttributeError:
            # 某些版本用 AddVolume，仍可運作（內部以順序索引）
            self.multi_volume.AddVolume(vol)

        # 記錄
        self.volumes.append(
            {
                "image": image_data,
                "volume": vol,
                "prop": prop,
                "color": list(cset) if cset else [],
                "opacity": list(oset) if oset else [],
            }
        )

        # 背景色與相機
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.ResetCameraClippingRange()
        self.store_initial_camera()
        self.render()

        return port  # 回傳此 volume 的索引

    def clear_volumes(self):
        """移除所有 volume 與輸入"""
        # 從 multi_volume 拿掉所有子 volume
        for i, v in enumerate(self.volumes):
            try:
                self.multi_volume.RemoveVolume(i)
            except AttributeError:
                # 有的版本沒有 RemoveVolume，就忽略；重建 mapper/multi_volume
                pass

        # 重新建立 mapper 與 multi_volume 以保險
        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.multi_volume = vtk.vtkMultiVolume()
        self.multi_volume.SetMapper(self.mapper)
        # 先清掉舊的，再加新的（避免重複）
        self.renderer.RemoveAllViewProps()
        self.renderer.AddViewProp(self.multi_volume)
        self.add_axes_indicator()

        self.volumes = []
        self.render()

        # 與舊 API 相容：單一 volume 的便利函式

    def setup_volume_data(self, data, spacing):
        """為相容舊程式：等同於清空後加入單一 volume"""

        with timer("渲染"):
            self.clear_volumes()
            if isinstance(data, list) and isinstance(spacing, list):
                for d, space in zip(data, spacing):
                    if d is not None and space is not None:
                        self.add_volume_data(d, space)
            elif data is not None and spacing is not None:
                self.add_volume_data(data, spacing)

        # 針對單一 volume 設定 transfer function

    def set_volume_transfer_functions(self, index, color_settings, opacity_settings):
        """
        index: volume 索引（add_volume_data 回傳值）
        color_settings: [(value, r, g, b), ...]
        opacity_settings: [(value, a), ...]
        """
        if not (0 <= index < len(self.volumes)):
            return
        v = self.volumes[index]
        v["color"] = list(color_settings or [])
        v["opacity"] = list(opacity_settings or [])
        self._apply_transfer_functions_to_property(v["prop"], v["color"], v["opacity"])
        self.render()

    def _apply_transfer_functions_to_property(
        self, prop, color_settings, opacity_settings
    ):
        # opacity
        pwf = vtk.vtkPiecewiseFunction()
        for val, op in opacity_settings:
            pwf.AddPoint(float(val), float(op))
        prop.SetScalarOpacity(pwf)
        # color
        ctf = vtk.vtkColorTransferFunction()
        for val, r, g, b in color_settings:
            ctf.AddRGBPoint(float(val), float(r), float(g), float(b))
        prop.SetColor(ctf)

    def render(self):
        """Render the scene"""
        if self.render_window:
            self.render_window.Render()
        else:
            print("Warning: Render window not available")

    def replace_camera(self):
        """依所有 volume 的總 bounds 擺相機"""
        if not self.volumes:
            return
        # 合併 bounds
        xmin, xmax, ymin, ymax, zmin, zmax = [
            np.inf,
            -np.inf,
            np.inf,
            -np.inf,
            np.inf,
            -np.inf,
        ]
        for v in self.volumes:
            b = v["image"].GetBounds()
            xmin = min(xmin, b[0])
            xmax = max(xmax, b[1])
            ymin = min(ymin, b[2])
            ymax = max(ymax, b[3])
            zmin = min(zmin, b[4])
            zmax = max(zmax, b[5])

        center = [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5]
        distance = max(xmax - xmin, ymax - ymin, zmax - zmin) * 2.0

        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(center[0], center[1], center[2] + distance)
        cam.SetFocalPoint(*center)
        cam.SetViewUp(0, 1, 0)
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def reset_camera(self):
        """Reset the camera to its initial angle with Y-axis upward"""
        camera = self.renderer.GetActiveCamera()
        if self.initial_camera is not None:
            camera.DeepCopy(self.initial_camera)
            camera.SetViewUp(0, 1, 0)
        else:
            self.renderer.ResetCamera()
            camera.SetViewUp(0, 1, 0)
            self.store_initial_camera()
        self.renderer.ResetCameraClippingRange()
        # print(f"Camera after setup/reset - Position: {camera.GetPosition()}, ViewUp: {camera.GetViewUp()}")
        self.render_window.Render()

    def store_initial_camera(self):
        """Store the current camera state as the initial camera state"""
        camera = self.renderer.GetActiveCamera()
        self.initial_camera = vtk.vtkCamera()
        self.initial_camera.DeepCopy(camera)

    def render(self):
        self.render_window.Render()

    def set_rotation_speed(self, speed):
        """Set the rotation speed of the renderer"""
        self.rotation_speed = speed

    def start_rotation(self):
        """Start rotating the renderer, ensuring a clean start"""
        if self.rotating:
            self.stop_rotation()

        if self.rotation_speed > 0:
            base_speed = 30.0
            timer_interval = 30

            def rotate_callback(obj, event):
                degrees_per_second = base_speed * self.rotation_speed
                angle_step = degrees_per_second * (timer_interval / 1000.0)
                camera = self.renderer.GetActiveCamera()
                camera.Azimuth(angle_step)
                self.renderer.ResetCameraClippingRange()
                self.render_window.Render()

            # Store the observer tag when adding it
            self.observer_tag = self.interactor.AddObserver(
                "TimerEvent", rotate_callback
            )
            self.timer_id = self.interactor.CreateRepeatingTimer(timer_interval)
            self.rotating = True
            if self.main_window:
                self.main_window.start_button.setEnabled(False)
                self.main_window.stop_button.setEnabled(True)

    def stop_rotation(self):
        """Stop rotating the renderer"""
        if self.rotating and self.interactor:
            self.rotating = False
            if self.timer_id is not None:
                self.interactor.DestroyTimer(self.timer_id)
                self.timer_id = None
            if self.observer_tag is not None:
                self.interactor.RemoveObserver(self.observer_tag)
                self.observer_tag = None
            if self.main_window:
                self.main_window.start_button.setEnabled(True)
                self.main_window.stop_button.setEnabled(False)

    def set_rotation_speed(self, speed):
        self.rotation_speed = speed

    def save_screenshot(self, filename):
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(f"{filename}.png")
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def record_rotation_video(self, filename, rotation_speed):
        """Record a 360° rotation video of the volume rendering"""
        if len(self.volumes) == 0:
            print("No volume data to record.", flush=True)
            return
        if rotation_speed <= 0:
            print(
                "Rotation speed is 0 or negative, no rotation will occur.", flush=True
            )
            return

        total_rotation = 360
        fps = 30
        base_speed = 60.0
        degrees_per_second = base_speed * rotation_speed
        recording_time = total_rotation / degrees_per_second
        total_frames = max(int(recording_time * fps), 30)  # Ensure at least 30 frames
        angle_step = total_rotation / total_frames

        # Print recording details as requested
        print(
            f"Recording 360° rotation: speed={rotation_speed:.1f}, "
            f"{total_frames} frames, {recording_time:.2f} seconds",
            flush=True,
        )

        if not filename.endswith(".mp4"):
            filename += ".mp4"
        writer = imageio.get_writer(filename, fps=fps, codec="libx264", quality=8)

        self.render_window.SetSize(1328, 960)
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetScale(1)

        camera = self.renderer.GetActiveCamera()
        initial_position = camera.GetPosition()

        print("Starting video recording...", flush=True)
        for frame in range(total_frames):
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

            # Optional: Print progress every 10% of the frames
            if total_frames > 10 and frame % (total_frames // 10) == 0:
                print(
                    f"Progress: {frame}/{total_frames} frames ({(frame/total_frames)*100:.0f}%)",
                    flush=True,
                )

        writer.close()
        camera.SetPosition(*initial_position)  # Reset camera to initial position
        self.render()

        print(f"Video recording completed. Saved as {filename}", flush=True)


if __name__ == "__main__":
    import torch
    from PyQt6.QtWidgets import QApplication
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    app = QApplication(sys.argv)
    vtk_widget = QVTKRenderWindowInteractor()
    renderer = VTKRenderer(vtk_widget)
    dummy_data = torch.zeros(128, 256, 256)
    renderer.setup_volume_data(dummy_data, (0.7, 0.7, 1.0))
    vtk_widget.Initialize()
    vtk_widget.Start()
    renderer.render()
    renderer.start_rotation()
    vtk_widget.show()
    app.exec()
