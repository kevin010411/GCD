import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QSpinBox

from src.gcd.presentation.qt.plugins.xai_family import XaiFamilyPluginPanel


class XaiFamilyPluginPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_method_switch_rebuilds_dynamic_parameters(self) -> None:
        panel = XaiFamilyPluginPanel(
            "perturbation",
            "Perturbation XAI",
            "Run perturbation methods.",
        )

        panel.set_method_options(
            [
                {
                    "id": "perturb_occlusion",
                    "name": "Occlusion",
                    "uses_layer_controls": False,
                    "uses_objective": True,
                    "parameters": [
                        {
                            "id": "block_size",
                            "label": "Block Size",
                            "kind": "int",
                            "default": 16,
                            "min": 1,
                            "max": 256,
                        },
                        {
                            "id": "stride",
                            "label": "Stride",
                            "kind": "int",
                            "default": 8,
                            "min": 1,
                            "max": 256,
                        },
                        {
                            "id": "baseline",
                            "label": "Baseline",
                            "kind": "float",
                            "default": 0.0,
                            "min": -10.0,
                            "max": 10.0,
                        },
                    ],
                }
            ],
            "perturb_occlusion",
        )

        self.assertFalse(panel.objective_row.isHidden())
        self.assertEqual(
            set(panel.selected_method_params()), {"block_size", "stride", "baseline"}
        )
        self.assertTrue(panel.layer_feature_group.isHidden())
        self.assertIsInstance(panel._parameter_widgets["block_size"], QSpinBox)
        self.assertEqual(panel.selected_method_params()["block_size"], 16)

    def test_method_option_refresh_preserves_parameter_values(self) -> None:
        panel = XaiFamilyPluginPanel(
            "perturbation",
            "Perturbation XAI",
            "Run perturbation methods.",
        )
        options = [
            {
                "id": "perturb_occlusion",
                "name": "Occlusion",
                "uses_layer_controls": False,
                "uses_objective": True,
                "parameters": [
                    {
                        "id": "block_size",
                        "label": "Block Size",
                        "kind": "int",
                        "default": 16,
                        "min": 1,
                        "max": 256,
                    }
                ],
            }
        ]
        panel.set_method_options(options, "perturb_occlusion")
        panel._parameter_widgets["block_size"].setValue(32)

        panel.set_method_options(options, "perturb_occlusion")

        self.assertEqual(panel.selected_method_params()["block_size"], 32)

    def test_saliency_method_hides_layer_feature_controls(self) -> None:
        panel = XaiFamilyPluginPanel("gradient", "Gradient XAI", "Run gradients.")

        panel.set_method_options(
            [
                {
                    "id": "saliency_map",
                    "name": "Saliency Map",
                    "uses_layer_controls": False,
                    "uses_objective": True,
                    "parameters": [],
                }
            ],
            "saliency_map",
        )
        panel.set_layer_options(["layer-a"], "layer-a", 8)

        self.assertTrue(panel.layer_feature_group.isHidden())
        self.assertFalse(panel.objective_row.isHidden())

    def test_layer_combo_enabled_before_feature_size_is_known(self) -> None:
        panel = XaiFamilyPluginPanel("gradient", "Gradient XAI", "Run gradients.")
        panel.set_method_options(
            [
                {
                    "id": "gradcam",
                    "name": "Grad-CAM",
                    "uses_layer_controls": True,
                    "uses_objective": True,
                    "parameters": [],
                }
            ],
            "gradcam",
        )

        panel.set_layer_options(["encoder 1", "decoder 1"], "decoder 1", 0)

        self.assertTrue(panel.layer_combo.isEnabled())
        self.assertFalse(panel.feature_widget.isEnabled())

    def test_feature_widget_enabled_after_feature_size_is_known(self) -> None:
        panel = XaiFamilyPluginPanel("gradient", "Gradient XAI", "Run gradients.")
        panel.set_method_options(
            [
                {
                    "id": "gradcam",
                    "name": "Grad-CAM",
                    "uses_layer_controls": True,
                    "uses_objective": True,
                    "parameters": [],
                }
            ],
            "gradcam",
        )

        panel.set_layer_options(["decoder 1"], "decoder 1", 8)

        self.assertTrue(panel.layer_combo.isEnabled())
        self.assertTrue(panel.feature_widget.isEnabled())


if __name__ == "__main__":
    unittest.main()
