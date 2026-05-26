import unittest

import numpy as np

from src.gcd.application.presenter import MainWindowPresenter
from src.gcd.domain import DataRange, DatasetInput, TransferFunction, VolumeRecord, XaiComputeResult


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
        self.enabled = True

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

    def setEnabled(self, enabled):
        self.enabled = enabled


class _FeatureWidget:
    def __init__(self) -> None:
        self.apply_requested = _Signal()
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = enabled


class _VolumeList:
    def __init__(self) -> None:
        self.selection_changed = _Signal()
        self.visibility_changed = _Signal()
        self.order_changed = _Signal()
        self.name_changed = _Signal()
        self.delete_requested = _Signal()
        self.items = []
        self.selected = None

    def set_volumes(self, items, selected):
        self.items = items
        self.selected = selected

    def selected_volume_id(self):
        return self.selected


class _TransferEditor:
    def __init__(self) -> None:
        self.transfer_function_changed = _Signal()
        self.transfer_function_change_finished = _Signal()
        self.load_requested = _Signal()
        self.save_requested = _Signal()
        self.export_png_requested = _Signal()
        self.export_png_calls = []
        self._export_path = "transfer_function.png"

    def blockSignals(self, *_args):
        pass

    def set_transfer_function(self, *_args):
        pass

    def choose_export_png_path(self):
        return self._export_path

    def export_png(self, *args):
        self.export_png_calls.append(args)


class _Workspace:
    def __init__(self) -> None:
        self.annotations_changed = _Signal()
        self.payloads = []
        self._overlay_status = ""
        self.replace_camera_calls = 0
        self.sync_camera_calls = 0
        self.transfer_calls = []
        self.render_calls = 0
        self.show_volumes_calls = []

    def set_workspace_payload(self, **kwargs):
        self.payloads.append(kwargs)

    def show_volumes(self, *args, **kwargs):
        self.show_volumes_calls.append((args, kwargs))

    def overlay_status_message(self):
        return self._overlay_status

    def set_volume_transfer_functions(self, *args, **kwargs):
        self.transfer_calls.append((args, kwargs))
        if kwargs.get("render"):
            self.render()

    def render(self):
        self.render_calls += 1

    def replace_camera(self):
        self.replace_camera_calls += 1

    def sync_camera_to_visible_volumes(self):
        self.sync_camera_calls += 1

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
        self.render_calls = 0
        self.transfer_calls = []

    def set_rotation_speed(self, _value):
        pass

    def show_volumes(self, *_args, **_kwargs):
        pass

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
        self.delete_volume_button = _Control()
        self.transfer_editor = _TransferEditor()
        self.layout_action_focus = _Control()
        self.layout_action_triple = _Control()
        self.layout_action_3d_only = _Control()
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
        self._selected_objective = "predicted_target_mask"
        self._selected_layer = "layer-a"
        self._feature_range = (0, 8)
        self._selected_model_path = "src/config/model/unet.py"
        self.method_options_calls = []
        self.objective_options_calls = []
        self.grad_dataset_options_calls = []
        self.perturb_dataset_options_calls = []
        self.transfer_volume_snapshots = []
        self.overlay_status_messages = []
        self.layer_options_calls = []
        self.feature_size_calls = []
        self.layer_control_calls = []

    def selected_class(self):
        return self._selected_class

    def selected_method(self):
        return self._selected_method

    def selected_objective(self):
        return self._selected_objective

    def selected_method_uses_layer_controls(self):
        return self._selected_method != "saliency_map"

    def selected_layer(self):
        return self._selected_layer

    def feature_range(self):
        return self._feature_range

    def choose_input_file(self):
        return "sample.nii.gz"

    def set_method_options(self, options, selected):
        self.method_options_calls.append((options, selected))

    def set_objective_options(self, options, selected):
        self.objective_options_calls.append((options, selected))

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

    def set_gradcam_layer_controls_enabled(self, enabled):
        self.layer_control_calls.append(enabled)
        self.layer_combo.setEnabled(enabled)
        self.feature_widget.setEnabled(enabled)

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
        return [
            {"id": "gradcam", "name": "Grad-CAM", "uses_layer_controls": True},
            {
                "id": "saliency_map",
                "name": "Saliency Map",
                "uses_layer_controls": False,
            },
        ]

    def list_objectives(self):
        return [
            {"id": "predicted_target_mask", "name": "Predicted Target Mask"},
            {"id": "target_logit_sum", "name": "Target Logit Sum"},
        ]

    def set_config(self, path):
        self.set_config_calls.append(path)

    def load_input(self, file_name, target_class, method=None):
        self.load_calls.append((file_name, target_class, method))
        return {
            "file_name": file_name,
            "dataset_input": DatasetInput(
                img0=None,
                img1=np.array([1.0], dtype=np.float32),
                origin_img=None,
                origin_meta={},
                origin_shape=(1,),
                img1_spacing=(1.5, 1.5, 2.0),
                display_metadata={
                    "vtk_origin": (1.0, 2.0, 3.0),
                    "affine": np.eye(4, dtype=np.float32),
                },
                layers={"layer-a": 8},
                file_name=file_name,
                target_class=target_class,
                active_method_id=method or "gradcam",
                active_objective_id="predicted_target_mask",
            ),
            "layer_names": ["layer-a"],
            "selected_layer": "layer-a",
            "method_options": [
                {"id": "gradcam", "name": "Grad-CAM", "uses_layer_controls": True},
                {
                    "id": "saliency_map",
                    "name": "Saliency Map",
                    "uses_layer_controls": False,
                },
            ],
            "selected_method": method or "gradcam",
            "objective_options": self.list_objectives(),
            "selected_objective": "predicted_target_mask",
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

    def compute_xai(self, dataset_input, request):
        self.compute_calls.append((dataset_input, request))
        data = np.array([1, 2, 3], dtype=np.float32)
        return XaiComputeResult(
            dataset_input=dataset_input,
            layer_names=("input",) if request.method == "saliency_map" else ("layer-a",),
            selected_layer="input" if request.method == "saliency_map" else "layer-a",
            method_options=({"id": request.method, "name": request.method},),
            selected_method=request.method,
            objective_options=(
                {"id": request.objective_id, "name": request.objective_id},
            ),
            selected_objective=request.objective_id,
            feature_size=1 if request.method == "saliency_map" else 8,
            volume=VolumeRecord(
                id="",
                dataset_id="",
                display_name=request.result_name,
                source="xai",
                method_id=request.method,
                data=data,
                data_range=DataRange(0.0, 1.0),
                transfer_function=TransferFunction.heatmap_preset(),
                spacing=(1.0, 1.0, 1.0),
                metadata={
                    "vtk_origin": (0.0, 0.0, 0.0),
                    "affine": np.eye(4, dtype=np.float32),
                },
                shape=(3,),
                source_base_item_id="",
                source_shape=(3,),
                source_spacing=(1.0, 1.0, 1.0),
                source_affine=np.eye(4, dtype=np.float32),
            ),
            volume_data_range=DataRange(0.0, 1.0),
        )


class _FakeTaskRunner:
    def submit(self, func, on_success, _on_error):
        on_success(func())


class _FakeErrorStore:
    def __init__(self) -> None:
        self.calls = []

    def save(self, *_args, **_kwargs):
        self.calls.append((_args, _kwargs))


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
        args, kwargs = view.workspace.show_volumes_calls[-1]
        np.testing.assert_array_equal(args[0][0], workflow.loaded_volume_data)
        self.assertEqual(kwargs["camera_policy"], "reset_if_first_or_empty")
        self.assertEqual(workflow.compute_calls, [])
        self.assertEqual(view.workspace.sync_camera_calls, 0)

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
            "sample.nii_unet_layer-a_class2",
            presenter.render_items[presenter.volume_order[1]]["display_name"],
        )

    def test_method_change_disables_layer_controls_for_saliency_map(self) -> None:
        view = _FakeView()
        view._selected_method = "saliency_map"
        workflow = _FakeWorkflow()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_FakeTaskRunner(),
            error_store=_FakeErrorStore(),
        )

        presenter.on_method_changed(0)

        self.assertFalse(view.layer_combo.enabled)
        self.assertFalse(view.feature_widget.enabled)

        view._selected_method = "gradcam"
        presenter.on_method_changed(0)

        self.assertTrue(view.layer_combo.enabled)
        self.assertTrue(view.feature_widget.enabled)

    def test_saliency_run_uses_input_layer_and_ignores_feature_range(self) -> None:
        view = _FakeView()
        view._selected_method = "saliency_map"
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

        request = workflow.compute_calls[-1][1]
        self.assertEqual(request.layer, "input")
        self.assertEqual((request.n1, request.n2), (0, 1))
        self.assertEqual(request.objective_id, "predicted_target_mask")
        self.assertIn(
            "sample.nii_unet_input_class2",
            presenter.render_items[presenter.volume_order[1]]["display_name"],
        )

    def test_gradcam_run_adds_new_result_each_time(self) -> None:
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
        first_result_id = presenter.volume_order[1]
        presenter.on_gradcam_run_requested()

        self.assertEqual(len(workflow.compute_calls), 2)
        self.assertEqual(len(presenter.volume_order), 3)
        self.assertEqual(presenter.volume_order[1], first_result_id)
        self.assertNotEqual(presenter.volume_order[2], first_result_id)
        self.assertEqual(
            presenter.datasets[dataset_id]["result_ids"],
            [first_result_id, presenter.volume_order[2]],
        )
        self.assertEqual(
            presenter.render_items[presenter.volume_order[2]]["display_name"],
            "sample.nii_unet_layer-a_class2_2",
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
        self.assertEqual(view.workspace.sync_camera_calls, 0)
        self.assertEqual(view.workspace.replace_camera_calls, 0)
        self.assertEqual(len(view.workspace.show_volumes_calls), 1)

    def test_delete_volume_removes_prediction_and_rerenders(self) -> None:
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
        result_id = presenter.volume_order[1]

        presenter.on_volume_delete_requested(result_id)

        self.assertEqual(presenter.volume_order, [presenter.datasets[dataset_id].base_volume_id])
        self.assertNotIn(result_id, presenter.render_items)
        self.assertGreater(len(view.workspace.show_volumes_calls), 0)
        _args, kwargs = view.workspace.show_volumes_calls[-1]
        self.assertEqual(kwargs["camera_policy"], "preserve")

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
        _args, kwargs = view.workspace.show_volumes_calls[-1]
        self.assertIn("render_settings", kwargs)

    def test_transfer_change_updates_renderer_without_rebuilding_volumes(self) -> None:
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
        show_count = len(view.workspace.show_volumes_calls)
        transfer_count = len(view.workspace.transfer_calls)

        presenter.on_transfer_function_changed(
            TransferFunction.heatmap_preset(), DataRange(0.0, 1.0)
        )

        self.assertEqual(len(view.workspace.show_volumes_calls), show_count)
        self.assertEqual(len(view.workspace.transfer_calls), transfer_count + 1)
        _args, kwargs = view.workspace.transfer_calls[-1]
        self.assertTrue(kwargs["render"])

    def test_transfer_drag_change_defers_workspace_payload_sync(self) -> None:
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
        payload_count = len(view.workspace.payloads)

        presenter.on_transfer_function_changed(
            TransferFunction.heatmap_preset(), DataRange(0.0, 1.0)
        )

        self.assertEqual(len(view.workspace.payloads), payload_count)

        presenter.on_transfer_function_change_finished(
            TransferFunction.heatmap_preset(), DataRange(0.0, 1.0)
        )

        self.assertEqual(len(view.workspace.payloads), payload_count + 1)

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

    def test_export_transfer_png_uses_current_transfer_state(self) -> None:
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
        transfer_function, data_range = presenter._current_transfer_state()

        presenter.on_export_transfer_png_requested()

        self.assertEqual(
            view.transfer_editor.export_png_calls,
            [("transfer_function.png", transfer_function, data_range)],
        )

    def test_export_transfer_png_cancel_does_not_export(self) -> None:
        view = _FakeView()
        view.transfer_editor._export_path = ""
        workflow = _FakeWorkflow()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_FakeTaskRunner(),
            error_store=_FakeErrorStore(),
        )

        presenter.on_export_transfer_png_requested()

        self.assertEqual(view.transfer_editor.export_png_calls, [])


if __name__ == "__main__":
    unittest.main()
