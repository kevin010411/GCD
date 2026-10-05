import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from mmengine import Config

from src.gcd.application.presenter import MainWindowPresenter
from src.gcd.application.workspace_store import WorkspaceDataStore
from src.gcd.domain import DataRange, TransferFunction, VolumeRecord, XaiComputeResult
from src.gcd.infrastructure.xai.engine.core_engine import GradCamEngine
from src.gcd.infrastructure.xai.runtime.model_identity import model_identity
from src.gcd.presentation.qt.workspace import SliceViewWidget, ViewerWorkspace, _sample_slice_with_affine
from src.gcd.presentation.qt.workspace_models import SliceOrientation
from tests.test_presenter import _FakeView, _FakeWorkflow, _FakeTaskRunner, _FakeErrorStore
from tests.test_workspace_store import _loaded_result

ROOT = Path(__file__).resolve().parents[1]


class ConfigLayerTests(unittest.TestCase):
    def test_model_identity_distinguishes_weights_but_ignores_explanation_choices(self):
        cfg = dict(model={"type": "UNet"}, ckpt="one.pth", size=128, stride=112, spacing=(1, 1, 1), permute=(0, 1, 2))
        key = model_identity(cfg)
        self.assertEqual(key, model_identity({**cfg, "default_layer": "other", "target_class": 2, "method": "scorecam"}))
        self.assertNotEqual(key, model_identity({**cfg, "ckpt": "two.pth"}))
        self.assertEqual(key, model_identity({**cfg, "spacing": (2, 2, 2)}))
        self.assertNotEqual(key, model_identity({**cfg, "model": {"type": "other"}}))


class LayerSwitchTests(unittest.TestCase):
    def test_switching_uncomputed_data_and_model_keeps_config_layers(self):
        view, workflow = _FakeView(), _FakeWorkflow()
        presenter = MainWindowPresenter(view, workflow, object(), object(), _FakeTaskRunner(), _FakeErrorStore())
        loaded = {**_loaded_result(), "layer_names": [], "selected_layer": "", "feature_size": 0}
        for dataset_id in ("a", "b"):
            presenter.data_store.add_loaded_dataset(dataset_id, loaded)
        for dataset_id in ("b", "a", "b"):
            presenter.selected_grad_dataset_id = dataset_id
            presenter._sync_xai_controls("gradient")
            self.assertEqual(view.layer_options_calls[-1], (["encoder 1", "decoder 1"], "decoder 1"))
        presenter.on_layer_changed("encoder 1")
        presenter._sync_xai_controls("gradient")
        self.assertEqual(view.layer_options_calls[-1], (["encoder 1", "decoder 1"], "encoder 1"))
        presenter.selected_grad_dataset_id = "a"
        presenter._sync_xai_controls("gradient")
        self.assertEqual(view.layer_options_calls[-1][1], "decoder 1")
        presenter.selected_grad_dataset_id = "b"
        presenter._sync_xai_controls("gradient")
        self.assertEqual(view.layer_options_calls[-1][1], "encoder 1")
        workflow.model_layer_metadata = dict(layer_names=["new-layer"], selected_layer="new-layer", model_key="new-model")
        presenter.on_model_changed(0)
        presenter._sync_xai_controls("gradient")
        self.assertEqual(view.layer_options_calls[-1], (["new-layer"], "new-layer"))
        self.assertEqual(view.feature_size_calls[-1], (0,))

    def test_model_change_is_deferred_while_xai_is_running(self):
        view, workflow = _FakeView(), _FakeWorkflow()
        presenter = MainWindowPresenter(view, workflow, object(), object(), _FakeTaskRunner(), _FakeErrorStore())
        presenter._set_xai_running(True, "gradient")
        self.assertFalse(view.model_combo.enabled)
        self.assertFalse(view.open_file_button.enabled)
        presenter.on_model_changed(0)
        self.assertEqual(workflow.set_config_calls, [])
        presenter._set_xai_running(False)
        self.assertTrue(view.model_combo.enabled)
        presenter.on_model_changed(0)
        self.assertEqual(len(workflow.set_config_calls), 1)


class PredictionStoreTests(unittest.TestCase):
    def test_each_dataset_model_has_one_prediction_and_many_heatmaps(self):
        store = WorkspaceDataStore()
        loaded = _loaded_result()
        for dataset_id in ("a", "b"):
            store.add_loaded_dataset(dataset_id, loaded)

        def volume(source, method, key):
            data = np.ones(3, dtype=np.uint8)
            return VolumeRecord("", "", "sample_model_prediction" if source == "prediction" else "heatmap", source, method, data, DataRange(0, 1), TransferFunction.label_preset(), (1, 1, 1), {}, (3,), "", (3,), (1, 1, 1), None, plugin_metadata={"model_key": key})

        def result(method, key):
            return XaiComputeResult(loaded["dataset_input"], ("layer-a",), "layer-a", (), method, (), "predicted_target_mask", 8, volume("xai", method, key), DataRange(0, 1), volume("prediction", "model_prediction", key), model_key=key)

        events = []
        store.subscribe(lambda event: events.append((event.name, len(store.volumes))))
        first_cam = store.upsert_xai_result("a", result("gradcam", "model-1"))
        prediction_id = store.create_prediction_volume_id("a", "model-1")
        store.rename_item(prediction_id, "my prediction")
        store.set_volume_visibility(prediction_id, False)
        custom_transfer = TransferFunction.heatmap_preset()
        store.volumes[prediction_id] = replace(store.volumes[prediction_id], transfer_function=custom_transfer)
        second_cam = store.upsert_xai_result("a", result("scorecam", "model-1"))
        third_cam = store.upsert_xai_result("a", result("saliency_map", "model-1"))
        self.assertEqual(len({first_cam, second_cam, third_cam}), 3)
        self.assertEqual(len([v for v in store.volumes.values() if v.source == "prediction"]), 1)
        self.assertEqual(store.volumes[prediction_id].display_name, "my prediction")
        self.assertFalse(store.volumes[prediction_id].visible)
        self.assertEqual(store.volumes[prediction_id].transfer_function, custom_transfer)
        self.assertEqual(store.volume_order.count(prediction_id), 1)
        self.assertEqual(store.datasets["a"].result_ids.count(prediction_id), 1)
        self.assertEqual(events[0], ("volume_upserted", 4))
        store.upsert_xai_result("a", result("gradcam", "model-2"))
        store.upsert_xai_result("b", result("gradcam", "model-1"))
        self.assertEqual(len([v for v in store.volumes.values() if v.source == "prediction"]), 3)
        store.delete_volume(prediction_id)
        store.upsert_xai_result("a", result("gradcam", "model-1"))
        self.assertEqual(len([v for v in store.volumes.values() if v.source == "prediction"]), 3)


class SliceInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_slider_and_spinbox_emit_one_change(self):
        widget = SliceViewWidget("test")
        widget.index_slider.setMaximum(20)
        widget.index_spinbox.setMaximum(20)
        changes = []
        widget.changed.connect(lambda *args: changes.append(args))
        widget.index_slider.setValue(3)
        self.assertEqual(changes, [("test", "slice_index", 3)])
        changes.clear()
        widget.index_spinbox.setValue(5)
        self.assertEqual(changes, [("test", "slice_index", 5)])
        widget.deleteLater()

    def make_workspace(self):
        workspace = ViewerWorkspace(initialize_3d_on_idle=False)
        workspace.apply_layout("triple_slice")
        workspace.set_workspace_payload(renderable_items=[dict(source="base", data=np.ones((32, 32, 32), dtype=np.float32))])
        self.addCleanup(workspace.deleteLater)
        return workspace

    def test_slice_change_does_not_update_unrelated_slices_or_3d(self):
        workspace = self.make_workspace()
        with patch.object(workspace, "_refresh_3d_view") as render, patch.object(workspace, "_refresh_renderer_annotations") as annotations, patch.object(workspace.slice_widgets["slice-2"], "set_image") as other:
            workspace.set_slice_index("slice-1", 5)
            render.assert_not_called()
            annotations.assert_not_called()
            other.assert_not_called()
            self.assertEqual(workspace.slice_widgets["slice-1"].index_spinbox.value(), 5)

    def test_rapid_drag_coalesces_and_keeps_final_slice(self):
        workspace = self.make_workspace()
        with patch.object(workspace, "refresh_slice_views", wraps=workspace.refresh_slice_views) as refresh:
            for index in range(1, 20):
                workspace.slice_widgets["slice-1"].index_slider.setValue(index)
            self.assertEqual(refresh.call_count, 0)
            QTest.qWait(40)
            self.assertEqual(refresh.call_count, 1)
            self.assertEqual(workspace.viewer_slice_states["slice-1"].slice_index, 19)
            self.assertEqual(workspace.slice_widgets["slice-1"].index_spinbox.value(), 19)

    def test_linked_slices_follow_without_refreshing_other_orientation(self):
        workspace = self.make_workspace()
        first = workspace.viewer_slice_states["slice-1"]
        second = workspace.viewer_slice_states["slice-2"]
        second.orientation = first.orientation
        first.is_linked = second.is_linked = True
        workspace.state.global_slice_link_mode = True
        with patch.object(workspace.slice_widgets["slice-2"], "set_image") as linked, patch.object(workspace.slice_widgets["slice-3"], "set_image") as unrelated:
            workspace.set_slice_index("slice-1", 7)
            self.assertEqual(second.slice_index, 7)
            linked.assert_called_once()
            unrelated.assert_not_called()

    def test_plane_sampling_matches_previous_torch_interpolation(self):
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(3)
        volume = rng.integers(-20, 20, (7, 8, 9), dtype=np.int16)
        coords = rng.uniform(-0.8, 8.3, (12, 13, 3)).astype(np.float32)
        world = np.concatenate((coords, np.ones((12, 13, 1), dtype=np.float32)), axis=-1)
        grid = np.stack([coords[..., 2] / 8 * 2 - 1, coords[..., 1] / 7 * 2 - 1, coords[..., 0] / 6 * 2 - 1], axis=-1)
        expected = F.grid_sample(torch.from_numpy(volume.astype(np.float32))[None, None], torch.from_numpy(grid)[None, None], align_corners=True)[0, 0, 0].numpy()
        actual = _sample_slice_with_affine(volume, world, np.eye(4, dtype=np.float32))
        np.testing.assert_allclose(actual, expected, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
