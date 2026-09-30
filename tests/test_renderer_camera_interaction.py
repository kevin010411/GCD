import unittest

from src.gcd.infrastructure.renderer import VtkVolumeRenderer


class CameraInteractionTests(unittest.TestCase):
    def test_mouse_interaction_targets_volume_renderer(self) -> None:
        try:
            from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor
        except ImportError:
            self.skipTest("VTK is not installed")

        window = vtkRenderWindow()
        window.SetSize(400, 400)
        interactor = vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        widget = type("Widget", (), {"GetRenderWindow": lambda self: window})()
        renderer = VtkVolumeRenderer(widget)

        renderer.camera_interactor_style.FindPokedRenderer(200, 200)
        self.assertIs(renderer.camera_interactor_style.GetCurrentRenderer(), renderer.renderer)
        self.assertIs(interactor.FindPokedRenderer(200, 200), renderer.plane_overlay_renderer)


if __name__ == "__main__":
    unittest.main()
