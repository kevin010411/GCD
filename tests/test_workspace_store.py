import unittest

import numpy as np

from src.gcd.application.workspace_store import WorkspaceDataStore
from src.gcd.domain import DataRange, DatasetInput, TransferFunction


def _loaded_result():
    volume = np.array([9, 8, 7], dtype=np.float32)
    return {
        "file_name": "sample.nii.gz",
        "dataset_input": DatasetInput(
            img0=None,
            img1=np.array([1.0], dtype=np.float32),
            origin_img=None,
            origin_meta={},
            origin_shape=(1,),
            img1_spacing=(1.5, 1.5, 2.0),
            display_metadata={"vtk_origin": (1.0, 2.0, 3.0)},
            layers={"layer-a": 8},
            file_name="sample.nii.gz",
            target_class=1,
            active_method_id="gradcam",
        ),
        "layer_names": ["layer-a"],
        "selected_layer": "layer-a",
        "feature_size": 8,
        "volume_data": volume,
        "spacing": (1.5, 1.5, 2.0),
        "display_metadata": {"vtk_origin": (1.0, 2.0, 3.0)},
        "volume_data_range": DataRange(0.0, 1.0),
        "volume_transfer_function": TransferFunction.base_preset(),
    }


def _xai_result(data):
    return {
        "dataset_input": DatasetInput(
            img0=None,
            img1=np.array([1.0], dtype=np.float32),
            origin_img=None,
            origin_meta={},
            origin_shape=(1,),
            img1_spacing=(1.5, 1.5, 2.0),
            display_metadata={"vtk_origin": (1.0, 2.0, 3.0)},
            layers={"layer-a": 8},
            file_name="sample.nii.gz",
            target_class=1,
            active_method_id="gradcam",
        ),
        "layer_names": ["layer-a"],
        "selected_layer": "layer-a",
        "method_options": [{"id": "gradcam", "name": "Grad-CAM"}],
        "selected_method": "gradcam",
        "feature_size": 8,
        "renderable_item": {
            "name": "sample_model_grad方法",
            "source": "xai",
            "method_id": "gradcam",
            "data": data,
            "data_range": DataRange(0.0, 1.0),
            "transfer_function": TransferFunction.heatmap_preset(),
            "spacing": (1.0, 1.0, 1.0),
            "metadata": {"vtk_origin": (0.0, 0.0, 0.0)},
            "shape": tuple(int(v) for v in data.shape),
        },
    }


class WorkspaceDataStoreTests(unittest.TestCase):
    def test_grad_and_render_share_dataset_and_volume_source(self) -> None:
        store = WorkspaceDataStore()
        store.add_loaded_dataset("dataset-1", _loaded_result())

        dataset_input = store.datasets["dataset-1"].input_state
        render_payloads = store.ordered_volume_payloads()

        self.assertEqual(dataset_input.file_name, "sample.nii.gz")
        np.testing.assert_array_equal(render_payloads[0], np.array([9, 8, 7], dtype=np.float32))

    def test_xai_result_adds_new_method_slot_each_run(self) -> None:
        store = WorkspaceDataStore()
        store.add_loaded_dataset("dataset-1", _loaded_result())

        first_id = store.upsert_xai_result(
            "dataset-1", _xai_result(np.array([1, 2, 3], dtype=np.float32))
        )
        second_id = store.upsert_xai_result(
            "dataset-1", _xai_result(np.array([4, 5, 6], dtype=np.float32))
        )

        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(store.volume_order), 3)
        self.assertEqual(store.volume_order[0], "dataset-1:base")
        self.assertEqual(store.datasets["dataset-1"]["result_ids"], [first_id, second_id])
        self.assertEqual(store.volumes[first_id].display_name, "sample_model_grad方法")
        self.assertEqual(store.volumes[second_id].display_name, "sample_model_grad方法_2")
        np.testing.assert_array_equal(
            store.ordered_volume_payloads()[2],
            np.array([4, 5, 6], dtype=np.float32),
        )

    def test_delete_prediction_volume_removes_result_and_updates_selection(self) -> None:
        store = WorkspaceDataStore()
        store.add_loaded_dataset("dataset-1", _loaded_result())
        result_id = store.upsert_xai_result(
            "dataset-1", _xai_result(np.array([1, 2, 3], dtype=np.float32))
        )

        deleted = store.delete_volume(result_id)

        self.assertTrue(deleted)
        self.assertNotIn(result_id, store.volumes)
        self.assertEqual(store.datasets["dataset-1"]["result_ids"], [])
        self.assertEqual(store.volume_order, ["dataset-1:base"])
        self.assertEqual(store.selected_transfer_volume_id, "dataset-1:base")

    def test_delete_base_volume_removes_dataset_and_predictions(self) -> None:
        store = WorkspaceDataStore()
        store.add_loaded_dataset("dataset-1", _loaded_result())
        result_id = store.upsert_xai_result(
            "dataset-1", _xai_result(np.array([1, 2, 3], dtype=np.float32))
        )

        deleted = store.delete_volume("dataset-1:base")

        self.assertTrue(deleted)
        self.assertEqual(store.datasets, {})
        self.assertNotIn("dataset-1:base", store.volumes)
        self.assertNotIn(result_id, store.volumes)
        self.assertEqual(store.volume_order, [])
        self.assertEqual(store.selected_transfer_volume_id, "")

    def test_store_mutations_emit_domain_events(self) -> None:
        store = WorkspaceDataStore()
        events = []
        store.subscribe(events.append)

        store.add_loaded_dataset("dataset-1", _loaded_result())
        store.set_volume_visibility("dataset-1:base", False)
        store.set_volume_order(["dataset-1:base"])

        self.assertEqual(
            [event.name for event in events],
            [
                "dataset_added",
                "volume_upserted",
                "volume_upserted",
                "volume_order_changed",
            ],
        )

    def test_delete_mutations_emit_domain_events(self) -> None:
        store = WorkspaceDataStore()
        events = []
        store.subscribe(events.append)
        store.add_loaded_dataset("dataset-1", _loaded_result())
        result_id = store.upsert_xai_result(
            "dataset-1", _xai_result(np.array([1, 2, 3], dtype=np.float32))
        )

        store.delete_volume(result_id)
        store.delete_volume("dataset-1:base")

        event_names = [event.name for event in events]
        self.assertIn("volume_deleted", event_names)
        self.assertIn("dataset_deleted", event_names)


if __name__ == "__main__":
    unittest.main()
