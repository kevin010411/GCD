import unittest

import torch

from src.gcd.infrastructure.xai.runtime.layer_hooks import XaiLayerHookManager


class _HookedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = torch.nn.Conv3d(1, 2, kernel_size=1)
        self.head = torch.nn.Conv3d(2, 2, kernel_size=1)
        self.xai_layer_targets = {"feature": "feature"}

    def forward(self, value):
        return self.head(self.feature(value))


class _TwoLayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Conv3d(1, 2, kernel_size=1)
        self.second = torch.nn.Conv3d(2, 2, kernel_size=1)
        self.xai_layer_targets = {"first": "first", "second": "second"}

    def forward(self, value):
        return self.second(self.first(value))


class XaiLayerHookManagerTests(unittest.TestCase):
    def test_captures_activation_and_gradient(self) -> None:
        model = _HookedModel()
        value = torch.ones((1, 1, 2, 2, 2), requires_grad=True)

        with XaiLayerHookManager(model) as hooks:
            output = model(value)
            layers = hooks.layers_by_name()
            output[:, 1].sum().backward()

        self.assertIn("feature", layers)
        self.assertEqual(tuple(layers["feature"].shape), (1, 2, 2, 2, 2))
        self.assertIsNotNone(layers["feature"].grad)

    def test_missing_module_path_reports_layer_and_path(self) -> None:
        model = _HookedModel()
        model.xai_layer_targets = {"bad": "missing.path"}

        with self.assertRaisesRegex(RuntimeError, "bad.*missing.path"):
            XaiLayerHookManager(model)

    def test_context_removes_handles(self) -> None:
        model = _HookedModel()

        with XaiLayerHookManager(model):
            self.assertGreater(len(model.feature._forward_hooks), 0)

        self.assertEqual(len(model.feature._forward_hooks), 0)

    def test_selected_layers_register_only_requested_hook(self) -> None:
        model = _TwoLayerModel()
        value = torch.ones((1, 1, 2, 2, 2), requires_grad=True)

        with XaiLayerHookManager(model, selected_layers=("second",)) as hooks:
            output = model(value)
            captured = hooks.layers_by_name()
            output.sum().backward()

        self.assertEqual(set(captured), {"second"})
        self.assertEqual(hooks.available_layer_names, ("first", "second"))
        self.assertEqual(len(model.first._forward_hooks), 0)
        self.assertEqual(len(model.second._forward_hooks), 0)

if __name__ == "__main__":
    unittest.main()
