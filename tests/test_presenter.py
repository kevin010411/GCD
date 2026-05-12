import unittest

import numpy as np

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
        self._items = []
        self._value = 0

    def blockSignals(self, *_args):
        pass

    def clear(self):
        self._items = []

    def addItem(self, text, data=None):
        self._items.append((text, data))

    def setCurrentIndex(self, *_args):
        pass

    def findData(self, data):
        for index, (_text, item_data) in enumerate(self._items):
            if item_data == data:
                return index
        return -1

    def currentData(self):
        return self._items[0][1] if self._items else None

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value


class _FeatureWidget:
    def __init__(self) -> None:
        self.apply_requested = _Signal()


class _VolumeList:
    def __init__(self) -> None:
        self.selection_changed = _Signal()
        self.visibility_changed = _Signal()
        self.order_changed = _Signal()
        self.name_changed = _Signal()
        self.items = []

    def set_volumes(self, items, selected):
        self.items = items
        self.selected = selected


class _TransferEditor:
    def __init__(self) -> None:
        self.transfer_function_changed = _Signal()
        self.load_requested = _Signal()
        self.save_requested = _Signal()

    def blockSignals(self, *_args):
        pass

    def set_transfer_function(self, *_args):
        pass


class _Workspace:
    def __init__(self) -> None:
        self.annotations_changed = _Signal()
        self.payloads = []
        self._overlay_status = ""
        self.replace_camera_calls = 0
        self.sync_camera_calls = 0
        self.store_initial_camera_calls = 0
        self.transfer_calls = []
        self.render_calls = 0
        self.show_volumes_calls = []

    def set_workspace_payload(self, **kwargs):
        self.payloads.append(kwargs)

    def show_volumes(self, *args):
        self.show_volumes_calls.append(args)

    def overlay_status_message(self):
        return self._overlay_status

    def set_volume_transfer_functions(self, *args, **kwargs):
        self.transfer_calls.append((args, kwargs))

    def render(self):
        self.render_calls += 1

    def replace_camera(self):
        self.replace_camera_calls += 1

    def sync_camera_to_visible_volumes(self):
        self.sync_camera_calls += 1

    def store_initial_camera(self):
        self.store_initial_camera_calls += 1

    def export_annotations(self):
        class _State:
            mode = type("Mode", (), {"value": "off"})()
            point_size = 8
            active_roi_box_id = None
            points = []
            boxes_2d = []
            boxes_3d = []

        return _State()

    def set_annotation_mode(self, *_args):
        pass

    def set_annotation_point_size(self, *_args):
        pass

    def set_active_roi_box(self, *_args):
        pass

    def clear_annotations(self):
        pass

    def delete_selected_annotation(self):
        pass

    def select_annotation(self, *_args):
        pass


class _FakeRenderer:
    def __init__(self) -> None:
        self.last_show_volumes = None
        self.render_calls = 0
        self.transfer_calls = []

    def set_rotation_speed(self, _value):
        pass

    def show_volumes(self, *args, **_kwargs):
        self.last_show_volumes = args

    def start_rotation(self):
        pass

    def set_volume_transfer_functions(self, *_args, **_kwargs):
        self.transfer_calls.append((_args, _kwargs))

    def render(self):
        self.render_calls += 1

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
        self.gradcam_dataset_combo = _Control()
        self.gradcam_run_button = _Control()
        self.perturbation_dataset_combo = _Control()
        self.perturbation_class_spinbox = type("Spin", (), {"value": lambda self: 4})()
        self.perturbation_method_combo = _Control()
        self.perturbation_block_size_spinbox = type("Spin", (), {"value": lambda self: 16})()
        self.perturbation_stride_spinbox = type("Spin", (), {"value": lambda self: 8})()
        self.perturbation_run_button = _Control()
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
        self.perturbation_plugin_button = _Control()
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
        self._selected_model_path = "src/config/model/unet.py"
        self.method_options_calls = []
        self.grad_dataset_options_calls = []
        self.perturb_dataset_options_calls = []
        self.transfer_volume_snapshots = []
        self.overlay_status_messages = []
        self.layer_options_calls = []
        self.feature_size_calls = []

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

    def set_gradcam_dataset_options(self, options, selected):
        self.grad_dataset_options_calls.append((options, selected))

    def set_perturbation_dataset_options(self, options, selected):
        self.perturb_dataset_options_calls.append((options, selected))

    def set_perturbation_method_options(self, *_args):
        pass

    def set_layer_options(self, *args):
        self.layer_options_calls.append(args)

    def set_feature_size(self, *args):
        self.feature_size_calls.append(args)

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

    def set_overlay_status_message(self, message):
        self.overlay_status_messages.append(message)

    def selected_model_path(self):
        return self._selected_model_path

    def selected_gradcam_dataset(self):
        return getattr(self, "_selected_grad_dataset", "")

    def selected_perturbation_dataset(self):
        return getattr(self, "_selected_perturb_dataset", "")

    def selected_perturbation_method(self):
        return "perturb_occlusion"


class _FakeWorkflow:
    def __init__(self) -> None:
        self.load_calls = []
        self.compute_calls = []
        self.set_config_calls = []
        self.loaded_volume_data = np.array([9, 8, 7], dtype=np.float32)
        self.engine = type(
            "Engine",
            (),
            {
                "volume_data": [0, 1, 2],
                "img1_spacing": (1.0, 1.0, 1.0),
                "display_metadata": {
                    "vtk_origin": (0.0, 0.0, 0.0),
                    "affine": np.eye(4, dtype=np.float32),
                },
            },
        )()

    def list_model_configs(self):
        return []

    def list_cam_methods(self):
        return [{"id": "gradcam", "name": "Grad-CAM"}]

    def set_config(self, path):
        self.set_config_calls.append(path)

    def load_input(self, file_name, target_class, method=None):
        self.load_calls.append((file_name, target_class, method))
        return {
            "file_name": file_name,
            "dataset_state": {
                "file_name": file_name,
                "target_class": target_class,
                "active_method_id": method or "gradcam",
            },
            "layer_names": ["layer-a"],
            "selected_layer": "layer-a",
            "method_options": [{"id": "gradcam", "name": "Grad-CAM"}],
            "selected_method": method or "gradcam",
            "feature_size": 8,
            "volume_data": self.loaded_volume_data,
            "spacing": (1.5, 1.5, 2.0),
            "display_metadata": {
                "vtk_origin": (1.0, 2.0, 3.0),
                "affine": np.eye(4, dtype=np.float32),
            },
                "xai_cache_key": "",
                "volume_data_range": DataRange(0.0, 1.0),
                "volume_transfer_function": TransferFunction.base_preset(),
                "messages": [],
        }

    def compute_dataset_result(self, dataset_state, **kwargs):
        self.compute_calls.append((dataset_state, kwargs))
        return {
            "dataset_state": {
                **dataset_state,
                "active_method_id": kwargs["method"],
            },
            "layer_names": ["layer-a"],
            "selected_layer": "layer-a",
            "method_options": [{"id": kwargs["method"], "name": kwargs["method"]}],
            "selected_method": kwargs["method"],
            "feature_size": 8,
            "renderable_item": {
                "name": kwargs["result_name"],
                "source": "xai",
                "method_id": kwargs["method"],
                "data": np.array([1, 2, 3], dtype=np.float32),
                "data_range": DataRange(0.0, 1.0),
                "transfer_function": TransferFunction.heatmap_preset(),
                "spacing": (1.0, 1.0, 1.0),
                "metadata": {
                    "vtk_origin": (0.0, 0.0, 0.0),
                    "affine": np.eye(4, dtype=np.float32),
                },
                "shape": (3,),
            },
        }


class _FakeTaskRunner:
    def submit(self, func, on_success, _on_error):
        on_success(func())


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
        self.assertEqual(len(presenter.datasets), 1)
        self.assertEqual(workflow.compute_calls, [])
        base_item = presenter.render_items[presenter.volume_order[0]]
        np.testing.assert_array_equal(base_item["data"], workflow.loaded_volume_data)
        self.assertEqual(base_item["spacing"], (1.5, 1.5, 2.0))
        self.assertEqual(base_item["metadata"]["vtk_origin"], (1.0, 2.0, 3.0))
        np.testing.assert_array_equal(view.workspace.show_volumes_calls[-1][0][0], workflow.loaded_volume_data)
        self.assertEqual(workflow.compute_calls, [])
        self.assertEqual(view.workspace.sync_camera_calls, 1)
        self.assertEqual(view.workspace.store_initial_camera_calls, 1)

    def test_gradcam_run_creates_transfer_item_after_load(self) -> None:
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
        dataset_id = presenter.dataset_order[0]
        view._selected_grad_dataset = dataset_id

        presenter.on_gradcam_run_requested()

        self.assertEqual(len(workflow.compute_calls), 1)
        self.assertEqual(len(presenter.volume_order), 2)
        self.assertIn(
            "sample.nii_unet_grad方法",
            presenter.render_items[presenter.volume_order[1]]["display_name"],
        )

    def test_visibility_change_renders_immediately(self) -> None:
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
        base_item_id = presenter.volume_order[0]

        presenter.on_volume_visibility_changed(base_item_id, False)

        self.assertGreater(view.workspace.render_calls, 0)
        self.assertEqual(view.workspace.sync_camera_calls, 1)
        self.assertEqual(view.workspace.replace_camera_calls, 0)

    def test_replace_camera_uses_workspace(self) -> None:
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

        presenter.on_replace_camera_requested()

        self.assertEqual(view.workspace.replace_camera_calls, 1)

    def test_render_current_items_uses_workspace_host(self) -> None:
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

        self.assertGreater(len(view.workspace.show_volumes_calls), 0)

    def test_transfer_item_rename_updates_presenter_state(self) -> None:
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
        base_item_id = presenter.volume_order[0]

        presenter.on_transfer_item_renamed(base_item_id, "Renamed Base")

        self.assertEqual(presenter.render_items[base_item_id]["display_name"], "Renamed Base")


if __name__ == "__main__":
    unittest.main()
