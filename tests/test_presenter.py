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
        self.renderer = type(
            "Renderer",
            (),
            {
                "preview_box": None,
                "preview_box_updates": 0,
                "_update_preview_box": lambda self: setattr(
                    self, "preview_box_updates", self.preview_box_updates + 1
                ),
                "render_calls": 0,
                "render": lambda self: setattr(
                    self, "render_calls", self.render_calls + 1
                ),
            },
        )()
        self.replace_camera_calls = 0
        self.sync_camera_calls = 0
        self.transfer_calls = []
        self.render_calls = 0
        self.show_volumes_calls = []
        self.update_volume_data_calls = []

    def set_workspace_payload(self, **kwargs):
        self.payloads.append(kwargs)

    def show_volumes(self, *args, **kwargs):
        self.show_volumes_calls.append((args, kwargs))

    def update_volume_data(self, *args, **kwargs):
        self.update_volume_data_calls.append((args, kwargs))
        return True

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
        self.rotating = False
        self.start_rotation_calls = 0
        self.stop_rotation_calls = 0

    def set_rotation_speed(self, _value):
        pass

    def show_volumes(self, *_args, **_kwargs):
        pass

    def start_rotation(self):
        self.rotating = True
        self.start_rotation_calls += 1

    def set_volume_transfer_functions(self, *_args, **_kwargs):
        self.transfer_calls.append((_args, _kwargs))

    def render(self):
        self.render_calls += 1

    def set_workspace_payload(self, **_kwargs):
        pass

    def stop_rotation(self):
        self.rotating = False
        self.stop_rotation_calls += 1

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
        self.perturb_answer_options_calls = []
        self.perturb_progress_calls = []
        self.perturb_progress_values = []
        self.perturb_preview_running_calls = []
        self.data_controls_enabled_calls = []
        self.camera_controls_enabled_calls = []
        self._selected_answer_data = ""
        self._selected_perturb_preview_enabled = False
        self._volume_save_path = "volume.nii.gz"

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

    def set_perturbation_answer_data_options(self, options, selected):
        self.perturb_answer_options_calls.append((options, selected))

    def set_perturbation_progress_running(self, running):
        self.perturb_progress_calls.append(running)

    def set_perturbation_progress(self, current, total):
        self.perturb_progress_values.append((current, total))

    def set_perturbation_preview_running(self, running):
        self.perturb_preview_running_calls.append(running)

    def set_data_controls_enabled(self, enabled):
        self.data_controls_enabled_calls.append(enabled)

    def set_camera_controls_enabled(self, enabled):
        self.camera_controls_enabled_calls.append(enabled)
        for control in (
            self.speed_slider,
            self.start_button,
            self.stop_button,
            self.replace_camera_button,
            self.import_camera_button,
            self.export_camera_button,
        ):
            control.setEnabled(enabled)

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

    def selected_perturbation_answer_data(self):
        return self._selected_answer_data

    def selected_perturbation_preview_enabled(self):
        return self._selected_perturb_preview_enabled

    def choose_volume_save_file(self, *_args):
        return self._volume_save_path


class _FakeWorkflow:
    def __init__(self) -> None:
        self.load_calls = []
        self.compute_calls = []
        self.set_config_calls = []
        self.model_configs = []
        self.model_layer_metadata = {
            "layer_names": ["encoder 1", "decoder 1"],
            "selected_layer": "decoder 1",
            "feature_size": 0,
        }
        self.loaded_volume_data = np.array([9, 8, 7], dtype=np.float32)
        self.loaded_origin_img = np.array([0, 1, 2], dtype=np.float32)
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
        return list(self.model_configs)

    def list_cam_methods(self):
        return [
            {"id": "gradcam", "name": "Grad-CAM", "uses_layer_controls": True},
            {
                "id": "saliency_map",
                "name": "Saliency Map",
                "uses_layer_controls": False,
            },
        ]

    def list_perturbation_methods(self):
        return [
            {
                "id": "perturb_occlusion",
                "name": "Occlusion",
                "uses_layer_controls": True,
            }
        ]

    def list_objectives(self):
        return [
            {"id": "predicted_target_mask", "name": "Predicted Target Mask"},
            {"id": "target_logit_sum", "name": "Target Logit Sum"},
        ]

    def set_config(self, path):
        self.set_config_calls.append(path)

    def list_current_model_layers(self):
        return dict(self.model_layer_metadata)

    def load_input(self, file_name, target_class, method=None):
        self.load_calls.append((file_name, target_class, method))
        return {
            "file_name": file_name,
            "dataset_input": DatasetInput(
                img0=None,
                img1=np.array([1.0], dtype=np.float32),
                origin_img=self.loaded_origin_img,
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
        progress_callback = (request.method_params or {}).get("_progress_callback")
        if callable(progress_callback):
            progress_callback({"current": 0, "total": 4})
            progress_callback({"current": 4, "total": 4})
        preview_callback = (request.method_params or {}).get("_preview_callback")
        if callable(preview_callback):
            preview_callback(
                {
                    "kind": "perturb_preview",
                    "data": np.array([5, 6, 7], dtype=np.float32),
                    "spacing": (1.0, 1.0, 1.0),
                    "metadata": {"volume_id": "perturb-preview"},
                    "preview_box": ((1, 2, 3), (4, 5, 6)),
                }
            )
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


class _DeferredTaskRunner:
    def __init__(self) -> None:
        self.pending = []

    def submit(self, func, on_success, on_error):
        self.pending.append((func, on_success, on_error))

    def succeed_next(self):
        func, on_success, _on_error = self.pending.pop(0)
        on_success(func())

    def fail_next(self, exc):
        _func, _on_success, on_error = self.pending.pop(0)
        on_error(exc)


class _ProgressTaskRunner:
    def submit(self, func, on_success, _on_error, on_progress=None):
        try:
            on_success(func(on_progress))
        except TypeError:
            on_success(func())


class _FakeVolumeService:
    def __init__(self) -> None:
        self.save_calls = []

    def save(self, volume, path):
        self.save_calls.append((volume, path))
        return path


class _FakeErrorStore:
    def __init__(self) -> None:
        self.calls = []

    def save(self, *_args, **_kwargs):
        self.calls.append((_args, _kwargs))


class PresenterMethodTests(unittest.TestCase):
    def test_model_change_refreshes_layer_options_before_forward(self) -> None:
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

        presenter.on_model_changed(0)

        self.assertEqual(workflow.set_config_calls, ["src/config/model/unet.py"])
        self.assertEqual(
            view.layer_options_calls[-1],
            (["encoder 1", "decoder 1"], "decoder 1"),
        )
        self.assertEqual(view.feature_size_calls[-1], (0,))

    def test_initialize_refreshes_layer_options_for_initial_model(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        workflow.model_configs = [
            {"name": "unet", "path": "src/config/model/unet.py"},
        ]
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_FakeTaskRunner(),
            error_store=_FakeErrorStore(),
        )

        presenter.initialize()

        self.assertEqual(workflow.set_config_calls, ["src/config/model/unet.py"])
        self.assertEqual(
            view.layer_options_calls[-1],
            (["encoder 1", "decoder 1"], "decoder 1"),
        )
        self.assertEqual(view.feature_size_calls[-1], (0,))

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

    def test_xai_run_ignores_second_request_while_one_is_running(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        task_runner = _DeferredTaskRunner()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=task_runner,
            error_store=_FakeErrorStore(),
        )
        presenter.on_open_file_requested()
        task_runner.succeed_next()
        dataset_id = presenter.dataset_order[0]
        view._selected_grad_dataset = dataset_id
        view._selected_perturb_dataset = dataset_id

        presenter.on_gradcam_run_requested()
        presenter.on_perturbation_run_requested()

        self.assertEqual(len(task_runner.pending), 1)
        self.assertEqual(len(workflow.compute_calls), 0)
        self.assertFalse(view.gradcam_run_button.enabled)
        self.assertFalse(view.perturbation_run_button.enabled)
        self.assertEqual(view.overlay_status_messages[-1], "XAI is already running.")
        self.assertNotIn(True, view.perturb_progress_calls)

        task_runner.succeed_next()

        self.assertEqual(len(workflow.compute_calls), 1)
        self.assertTrue(view.gradcam_run_button.enabled)
        self.assertTrue(view.perturbation_run_button.enabled)

    def test_perturbation_run_toggles_progress_bar(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        task_runner = _DeferredTaskRunner()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=task_runner,
            error_store=_FakeErrorStore(),
        )
        presenter.on_open_file_requested()
        task_runner.succeed_next()
        dataset_id = presenter.dataset_order[0]
        view._selected_perturb_dataset = dataset_id

        presenter.on_perturbation_run_requested()

        self.assertEqual(view.perturb_progress_calls[-1], True)

        task_runner.succeed_next()

        self.assertEqual(view.perturb_progress_calls[-1], False)

    def test_perturbation_run_updates_progress_values(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_ProgressTaskRunner(),
            error_store=_FakeErrorStore(),
        )
        presenter.on_open_file_requested()
        view._selected_perturb_dataset = presenter.dataset_order[0]

        presenter.on_perturbation_run_requested()

        self.assertEqual(view.perturb_progress_values, [(0, 4), (4, 4)])

    def test_perturbation_run_passes_selected_data_volume_as_answer(self) -> None:
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
        base_volume_id = presenter.datasets[dataset_id].base_volume_id
        view._selected_perturb_dataset = dataset_id
        view._selected_answer_data = base_volume_id

        presenter.on_perturbation_run_requested()

        request = workflow.compute_calls[-1][1]
        self.assertIs(
            request.method_params["answer_data"],
            workflow.loaded_origin_img,
        )
        self.assertEqual(request.method_params["answer_volume_id"], base_volume_id)

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

    def test_save_volume_uses_selected_store_volume(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        volume_service = _FakeVolumeService()
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_FakeTaskRunner(),
            error_store=_FakeErrorStore(),
            volume_service=volume_service,
        )
        presenter.on_open_file_requested()
        volume_id = presenter.volume_order[0]

        presenter.on_volume_save_requested(volume_id)

        self.assertEqual(len(volume_service.save_calls), 1)
        saved_volume, path = volume_service.save_calls[0]
        self.assertEqual(saved_volume.id, volume_id)
        self.assertEqual(path, "volume.nii.gz")

    def test_perturb_preview_locks_data_and_restores_workspace(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        view._selected_perturb_preview_enabled = True
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=_ProgressTaskRunner(),
            error_store=_FakeErrorStore(),
        )
        presenter.on_open_file_requested()
        dataset_id = presenter.dataset_order[0]
        view._selected_perturb_dataset = dataset_id

        presenter.on_perturbation_run_requested()

        request = workflow.compute_calls[-1][1]
        self.assertIn("_preview_callback", request.method_params)
        self.assertIn("_pause_controller", request.method_params)
        self.assertIn(False, view.data_controls_enabled_calls)
        self.assertIn(True, view.data_controls_enabled_calls)
        self.assertEqual(view.camera_controls_enabled_calls[0], False)
        self.assertEqual(view.camera_controls_enabled_calls[-1], True)
        self.assertIn(True, view.perturb_preview_running_calls)
        self.assertIn(False, view.perturb_preview_running_calls)
        preview_call = next(
            call
            for call in view.workspace.show_volumes_calls
            if call[0][2][0].get("volume_id") == "perturb-preview"
        )
        np.testing.assert_array_equal(
            preview_call[0][0][0], np.array([5, 6, 7], dtype=np.float32)
        )
        preview_range = DataRange.from_data(
            [np.array([5, 6, 7], dtype=np.float32)], method="minmax"
        )
        expected_colors, expected_opacities = (
            TransferFunction.base_preset().renderer_points(preview_range)
        )
        self.assertEqual(
            preview_call[1]["render_settings"][0]["color"], expected_colors
        )
        self.assertEqual(
            preview_call[1]["render_settings"][0]["opacity"], expected_opacities
        )
        self.assertIsNone(view.workspace.renderer.preview_box)
        self.assertGreaterEqual(view.workspace.renderer.preview_box_updates, 2)

    def test_perturb_preview_camera_controls_are_enabled_only_while_paused(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        task_runner = _DeferredTaskRunner()
        view._selected_perturb_preview_enabled = True
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=task_runner,
            error_store=_FakeErrorStore(),
        )
        presenter.on_open_file_requested()
        task_runner.succeed_next()
        view._selected_perturb_dataset = presenter.dataset_order[0]

        presenter.on_perturbation_run_requested()

        self.assertEqual(view.camera_controls_enabled_calls[-1], False)
        presenter.on_replace_camera_requested()
        self.assertEqual(view.workspace.replace_camera_calls, 0)
        presenter.on_perturbation_preview_pause_toggled(True)
        self.assertEqual(view.camera_controls_enabled_calls[-1], True)
        presenter.on_replace_camera_requested()
        self.assertEqual(view.workspace.replace_camera_calls, 1)
        presenter.on_perturbation_preview_pause_toggled(False)
        self.assertEqual(view.camera_controls_enabled_calls[-1], False)

        task_runner.succeed_next()

        self.assertEqual(view.camera_controls_enabled_calls[-1], True)

    def test_perturb_preview_stops_and_restores_existing_rotation(self) -> None:
        view = _FakeView()
        workflow = _FakeWorkflow()
        task_runner = _DeferredTaskRunner()
        view._selected_perturb_preview_enabled = True
        presenter = MainWindowPresenter(
            view,
            workflow,
            transfer_service=object(),
            annotation_service=object(),
            task_runner=task_runner,
            error_store=_FakeErrorStore(),
        )
        presenter.on_open_file_requested()
        task_runner.succeed_next()
        view.renderer.rotating = True
        view.renderer.start_rotation_calls = 0
        view.renderer.stop_rotation_calls = 0
        view._selected_perturb_dataset = presenter.dataset_order[0]

        presenter.on_perturbation_run_requested()

        self.assertFalse(view.renderer.rotating)
        self.assertEqual(view.renderer.stop_rotation_calls, 1)

        task_runner.succeed_next()

        self.assertTrue(view.renderer.rotating)
        self.assertEqual(view.renderer.start_rotation_calls, 1)

    def test_perturb_preview_updates_existing_volume_without_rebuilding_scene(self) -> None:
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
        presenter.selected_perturbation_dataset_id = presenter.dataset_order[0]

        presenter._show_perturbation_preview(
            {
                "kind": "perturb_preview",
                "data": np.array([1, 2, 3], dtype=np.float32),
                "spacing": (1.0, 1.0, 1.0),
                "metadata": {"volume_id": "perturb-preview"},
                "preview_box": ((0, 0, 0), (1, 1, 1)),
            }
        )
        presenter._show_perturbation_preview(
            {
                "kind": "perturb_preview",
                "data": np.array([4, 5, 6], dtype=np.float32),
                "spacing": (1.0, 1.0, 1.0),
                "metadata": {"volume_id": "perturb-preview"},
                "preview_box": ((1, 1, 1), (2, 2, 2)),
            }
        )

        preview_rebuilds = [
            call
            for call in view.workspace.show_volumes_calls
            if call[0][2][0].get("volume_id") == "perturb-preview"
        ]
        self.assertEqual(len(preview_rebuilds), 1)
        self.assertEqual(len(view.workspace.update_volume_data_calls), 1)
        np.testing.assert_array_equal(
            view.workspace.update_volume_data_calls[0][0][1],
            np.array([4, 5, 6], dtype=np.float32),
        )
        self.assertEqual(view.workspace.renderer.preview_box, ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)))

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
