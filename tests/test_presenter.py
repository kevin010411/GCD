import unittest

from src.gcd.application.presenter import MainWindowPresenter
from src.gcd.domain import DataRange, TransferFunction


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)


class _Control:
    def __init__(self) -> None:
        self.clicked = _Signal()
        self.triggered = _Signal()
        self.currentIndexChanged = _Signal()
        self.currentTextChanged = _Signal()
        self.valueChanged = _Signal()
        self.toggled = _Signal()


class _FeatureWidget:
    def __init__(self) -> None:
        self.apply_requested = _Signal()


class _VolumeList:
    def __init__(self) -> None:
        self.selection_changed = _Signal()
        self.visibility_changed = _Signal()
        self.order_changed = _Signal()


class _TransferEditor:
    def __init__(self) -> None:
        self.transfer_function_changed = _Signal()
        self.load_requested = _Signal()
        self.save_requested = _Signal()


class _Workspace:
    def __init__(self) -> None:
        self.annotations_changed = _Signal()


class _FakeRenderer:
    def set_rotation_speed(self, _value):
        pass

    def show_volumes(self, *_args, **_kwargs):
        pass

    def start_rotation(self):
        pass

    def set_volume_transfer_functions(self, *_args, **_kwargs):
        pass

    def render(self):
        pass

    def set_workspace_payload(self, **_kwargs):
        pass

    def stop_rotation(self):
        pass

    def clear_volumes(self):
        pass

    def store_initial_camera(self):
        pass


class _FakeView:
    def __init__(self) -> None:
        self.model_combo = _Control()
        self.open_file_button = _Control()
        self.replace_camera_button = _Control()
        self.speed_slider = _Control()
        self.start_button = _Control()
        self.stop_button = _Control()
        self.import_camera_button = _Control()
        self.export_camera_button = _Control()
        self.layer_combo = _Control()
        self.method_combo = _Control()
        self.class_spinbox = _Control()
        self.save_screenshot_button = _Control()
        self.record_video_button = _Control()
        self.volume_list = _VolumeList()
        self.transfer_editor = _TransferEditor()
        self.layout_action_focus = _Control()
        self.layout_action_triple = _Control()
        self.layout_action_quad = _Control()
        self.layout_action_compare = _Control()
        self.gradcam_plugin_button = _Control()
        self.transfer_plugin_button = _Control()
        self.camera_plugin_button = _Control()
        self.roi_plugin_button = _Control()
        self.roi_mode_combo = _Control()
        self.roi_point_size_slider = _Control()
        self.roi_point_size_spinbox = _Control()
        self.roi_box_combo = _Control()
        self.roi_export_button = _Control()
        self.roi_import_button = _Control()
        self.roi_clear_all_button = _Control()
        self.roi_delete_selected_button = _Control()
        self.roi_annotation_list = _Control()
        self.inspector_toggle_button = _Control()
        self.feature_widget = _FeatureWidget()
        self.workspace = _Workspace()
        self.renderer = _FakeRenderer()
        self._selected_class = 2
        self._selected_method = "gradcam"
        self._selected_layer = "layer-a"
        self._feature_range = (0, 8)
        self.method_options_calls = []

    def selected_class(self):
        return self._selected_class

    def selected_method(self):
        return self._selected_method

    def selected_layer(self):
        return self._selected_layer

    def feature_range(self):
        return self._feature_range

    def choose_input_file(self):
        return "sample.nii.gz"

    def set_file_name(self, _name):
        pass

    def set_method_options(self, options, selected):
        self.method_options_calls.append((options, selected))

    def set_layer_options(self, *_args):
        pass

    def set_feature_size(self, *_args):
        pass

    def set_rotation_running(self, *_args):
        pass

    def set_model_options(self, *_args):
        pass

    def set_rotation_speed_label(self, *_args):
        pass

    def set_active_plugin(self, *_args):
        pass

    def toggle_inspector(self, *_args):
        pass


class _FakeWorkflow:
    def __init__(self) -> None:
        self.load_calls = []

    def list_model_configs(self):
        return []

    def list_cam_methods(self):
        return [{"id": "gradcam", "name": "Grad-CAM"}]

    def load_input(self, file_name, target_class, method=None):
        self.load_calls.append((file_name, target_class, method))
        return {
            "file_name": file_name,
            "layer_names": ["layer-a"],
            "selected_layer": "layer-a",
            "method_options": [{"id": "gradcam", "name": "Grad-CAM"}],
            "selected_method": method or "gradcam",
            "feature_size": 8,
            "render_request": {},
            "cam_data_range": DataRange(0.0, 1.0),
            "volume_data_range": DataRange(0.0, 1.0),
            "cam_transfer_function": TransferFunction.heatmap_preset(),
            "volume_transfer_function": TransferFunction.base_preset(),
            "messages": [],
        }

    def compute_cam(self, **_kwargs):
        raise AssertionError("compute_cam should not be called in this test.")


class _FakeTaskRunner:
    def submit(self, func, on_success, _on_error):
        func()


class _FakeErrorStore:
    def save(self, *_args, **_kwargs):
        pass


class PresenterMethodTests(unittest.TestCase):
    def test_open_file_passes_selected_method_to_workflow(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_FakeTaskRunner(),
            error_store=_FakeErrorStore(),
        )

        presenter.on_open_file_requested()

        self.assertEqual(workflow.load_calls, [("sample.nii.gz", 2, "gradcam")])

    def test_method_change_reloads_current_file_with_selected_method(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_FakeTaskRunner(),
            error_store=_FakeErrorStore(),
        )
        presenter.current_file = "existing.nii.gz"

        presenter.on_method_changed(0)

        self.assertEqual(workflow.load_calls, [("existing.nii.gz", 2, "gradcam")])


if __name__ == "__main__":
    unittest.main()
