import unittest
from unittest.mock import patch

import torch

from src.gcd.infrastructure.xai.runtime.model_runtime_loader import (
    ModelLoadStateDictError,
    ModelRuntimeLoader,
)


class _Cfg:
    model = object()
    ckpt = "checkpoint.pt"
    filename = "config.py"


class _Model(torch.nn.Module):
    def __init__(self, *, missing=None, unexpected=None) -> None:
        super().__init__()
        self.missing = list(missing or [])
        self.unexpected = list(unexpected or [])
        self.loaded = False

    def load_state_dict(self, _state_dict, strict=False):
        self.loaded = True
        return self.missing, self.unexpected


class _ErrorStore:
    def __init__(self) -> None:
        self.payloads = []

    def save_json(self, payload, suffix):
        self.payloads.append((payload, suffix))
        return "model_load_error.json"


class ModelRuntimeLoaderTests(unittest.TestCase):
    def test_load_builds_model_loads_checkpoint_and_sets_eval(self) -> None:
        model = _Model()
        loader = ModelRuntimeLoader(build_model=lambda _cfg: model)

        with (
            patch(
                "src.gcd.infrastructure.xai.runtime.model_runtime_loader.os.path.exists",
                return_value=True,
            ),
            patch("torch.load", return_value={"state_dict": {"weight": torch.ones(1)}}),
        ):
            runtime = loader.load(_Cfg())

        self.assertIs(runtime.model, model)
        self.assertEqual(str(runtime.device), "cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(runtime.checkpoint_path, "checkpoint.pt")
        self.assertTrue(model.loaded)
        self.assertFalse(model.training)

    def test_load_raises_when_checkpoint_is_missing(self) -> None:
        loader = ModelRuntimeLoader(build_model=lambda _cfg: _Model())

        with (
            patch(
                "src.gcd.infrastructure.xai.runtime.model_runtime_loader.os.path.exists",
                return_value=False,
            ),
            self.assertRaises(FileNotFoundError),
        ):
            loader.load(_Cfg())

    def test_load_state_dict_mismatch_is_saved_to_error_store(self) -> None:
        error_store = _ErrorStore()
        loader = ModelRuntimeLoader(
            build_model=lambda _cfg: _Model(missing=["a"], unexpected=["b"]),
            error_store=error_store,
        )

        with (
            patch(
                "src.gcd.infrastructure.xai.runtime.model_runtime_loader.os.path.exists",
                return_value=True,
            ),
            patch("torch.load", return_value={}),
            self.assertRaises(ModelLoadStateDictError) as raised,
        ):
            loader.load(_Cfg())

        self.assertEqual(raised.exception.error_file, "model_load_error.json")
        self.assertEqual(len(error_store.payloads), 1)
        payload, suffix = error_store.payloads[0]
        self.assertEqual(suffix, "model_load_error")
        self.assertEqual(payload["missing_keys"], ["a"])
        self.assertEqual(payload["unexpected_keys"], ["b"])


if __name__ == "__main__":
    unittest.main()
