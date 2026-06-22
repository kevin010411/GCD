import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.gcd.application.perturbation_preview import PerturbationPreviewAdapter
from src.gcd.application.workspace_store import WorkspaceDataStore
from src.gcd.domain import DataRange, TransferFunction, VolumeRecord
from src.gcd.domain.workspace_data import DatasetInput, DatasetRecord
from src.gcd.infrastructure.model_input_preprocessor import (
    ModelInputPreprocessConfig,
    ModelInputPreprocessor,
)
from src.gcd.infrastructure.volume_loading import (
    VolumeLoadConfig,
    VolumeLoadingService,
)


class _TensorWithMeta:
    def __init__(self, data, meta=None) -> None:
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.meta = dict(meta or {})

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, item):
        return self.data[item]


class _Transform:
    def __init__(self, result) -> None:
        self.result = result

    def __call__(self, *_args, **_kwargs):
        return self.result


class RefactoredServiceTests(unittest.TestCase):
    def test_volume_loading_service_returns_display_volume_and_metadata(self) -> None:
        affine = np.diag([2.0, 3.0, 4.0, 1.0]).astype("float32")
        origin = torch.zeros((2, 3, 4), dtype=torch.float32)
        img0 = _TensorWithMeta(
            torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4),
            {"affine": affine},
        )

        with (
            patch("monai.transforms.LoadImage", return_value=_Transform((origin, {}))),
            patch("monai.transforms.EnsureChannelFirst", return_value=_Transform(img0)),
        ):
            loaded = VolumeLoadingService().load(
                "sample.nii.gz",
                VolumeLoadConfig(
                    spacing=(0.7, 0.7, 1.0),
                    permute=(1, 2, 0),
                    default_layer="layer-a",
                ),
            )

        self.assertEqual(loaded.file_name, "sample.nii.gz")
        self.assertEqual(loaded.origin_shape, (1, 2, 3, 4))
        self.assertEqual(loaded.display_volume.shape, torch.Size((3, 4, 2)))
        self.assertEqual(loaded.display_metadata["source_shape"], (2, 3, 4))
        self.assertEqual(loaded.display_metadata["display_permute"], (1, 2, 0))
        self.assertEqual(loaded.default_layers, {"layer-a": 1})
        self.assertEqual(loaded.messages, [])

    def test_model_input_preprocessor_pads_and_builds_model_metadata(self) -> None:
        img0 = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        spaced = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        spaced.meta = {"affine": np.eye(4, dtype=np.float32)}

        with (
            patch("monai.transforms.Spacing", return_value=_Transform(spaced)),
            patch("monai.transforms.SpatialPad", return_value=_Transform(spaced)),
            patch(
                "monai.transforms.ScaleIntensityRange",
                return_value=_Transform(torch.ones((1, 4, 4, 2), dtype=torch.float32)),
            ),
        ):
            result = ModelInputPreprocessor().preprocess(
                img0=img0,
                origin_img=None,
                origin_meta={},
                origin_shape=(1, 1, 1, 1),
                config=ModelInputPreprocessConfig(
                    size=2,
                    stride=2,
                    spacing=(1.0, 2.0, 3.0),
                    permute=(1, 2, 0),
                ),
            )

        self.assertEqual(result.img1.shape, torch.Size((1, 4, 4, 2)))
        self.assertEqual(result.img1_spacing, (2.0, 3.0, 1.0))
        self.assertEqual(result.messages, ["info: image is zero padded"])
        self.assertEqual(result.model_input_metadata["source_shape"], (1, 1, 1))
        self.assertEqual(result.model_input_metadata["display_permute"], (1, 2, 0))

    def test_perturbation_preview_adapter_handles_lifecycle_and_update(self) -> None:
        class _View:
            def __init__(self) -> None:
                self.workspace = type(
                    "Workspace",
                    (),
                    {
                        "show_volumes_calls": [],
                        "update_volume_data_calls": [],
                        "transfer_calls": [],
                        "renderer": type(
                            "Renderer",
                            (),
                            {
                                "preview_box": None,
                                "preview_updates": 0,
                                "render_calls": 0,
                                "_update_preview_box": lambda self: setattr(
                                    self, "preview_updates", self.preview_updates + 1
                                ),
                                "render": lambda self: setattr(
                                    self, "render_calls", self.render_calls + 1
                                ),
                            },
                        )(),
                        "show_volumes": lambda self, *args, **kwargs: self.show_volumes_calls.append((args, kwargs)),
                        "update_volume_data": lambda self, *args, **kwargs: self.update_volume_data_calls.append((args, kwargs)) or True,
                        "set_volume_transfer_functions": lambda self, *args, **kwargs: self.transfer_calls.append((args, kwargs)),
                    },
                )()
                self.renderer = type(
                    "MainRenderer",
                    (),
                    {
                        "rotating": True,
                        "stop_calls": 0,
                        "start_calls": 0,
                        "stop_rotation": lambda self: (
                            setattr(self, "rotating", False),
                            setattr(self, "stop_calls", self.stop_calls + 1),
                        ),
                        "start_rotation": lambda self: (
                            setattr(self, "rotating", True),
                            setattr(self, "start_calls", self.start_calls + 1),
                        ),
                    },
                )()
                self.preview_running = []
                self.rotation_running = []

            def set_perturbation_preview_running(self, running):
                self.preview_running.append(running)

            def set_rotation_running(self, running):
                self.rotation_running.append(running)

        view = _View()
        store = WorkspaceDataStore()
        dataset_input = DatasetInput(
            img0=None,
            img1=None,
            origin_img=None,
            origin_meta={},
            origin_shape=(1,),
            img1_spacing=(1.0, 1.0, 1.0),
            display_metadata={},
            layers={},
            file_name="sample.nii.gz",
            target_class=1,
            active_method_id="perturb_occlusion",
        )
        store.datasets["dataset-1"] = DatasetRecord(
            id="dataset-1",
            name="sample",
            file_name="sample.nii.gz",
            input_state=dataset_input,
            layer_names=(),
            selected_layer="",
            feature_size=0,
            base_volume_id="dataset-1:base",
            result_ids=(),
            base_shape=(2, 2, 2),
            base_spacing=(1.0, 1.0, 1.0),
            display_metadata={},
        )
        store.volumes["dataset-1:base"] = VolumeRecord(
            id="dataset-1:base",
            dataset_id="dataset-1",
            display_name="sample",
            source="base",
            method_id="base",
            data=torch.ones((2, 2, 2), dtype=torch.float32),
            data_range=DataRange(0.0, 1.0),
            transfer_function=TransferFunction.base_preset(),
            spacing=(1.0, 1.0, 1.0),
            metadata={},
            shape=(2, 2, 2),
            source_base_item_id="dataset-1:base",
            source_shape=(2, 2, 2),
            source_spacing=(1.0, 1.0, 1.0),
            source_affine=None,
        )
        data_controls = []
        camera_controls = []
        render_calls = []
        adapter = PerturbationPreviewAdapter(
            view=view,
            data_store=store,
            render_current_items=lambda **kwargs: render_calls.append(kwargs),
            set_data_controls_enabled=data_controls.append,
            set_camera_controls_enabled=camera_controls.append,
            selected_dataset_id=lambda: "dataset-1",
        )

        adapter.begin_if_enabled(True)
        adapter.handle_payload(
            {
                "data": torch.ones((2, 2, 2), dtype=torch.float32),
                "spacing": (1.0, 1.0, 2.0),
                "metadata": {"volume_id": "preview"},
                "preview_box": ((0, 1, 2), (3, 4, 5)),
            }
        )
        adapter.handle_payload(
            {
                "data": torch.zeros((2, 2, 2), dtype=torch.float32),
                "spacing": (1.0, 1.0, 2.0),
                "metadata": {"volume_id": "preview"},
            }
        )
        adapter.pause_toggled(True)
        adapter.finish()

        self.assertEqual(data_controls, [False, True])
        self.assertEqual(camera_controls, [False, True, True])
        self.assertEqual(view.preview_running, [True])
        self.assertEqual(len(view.workspace.show_volumes_calls), 1)
        self.assertEqual(len(view.workspace.update_volume_data_calls), 1)
        self.assertEqual(view.workspace.renderer.preview_box, None)
        self.assertEqual(render_calls, [{"camera_policy": "preserve"}])
        self.assertFalse(adapter.enabled)
        self.assertFalse(adapter.active)


if __name__ == "__main__":
    unittest.main()
