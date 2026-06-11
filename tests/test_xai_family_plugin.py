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
                    "uses_layer_controls": True,
                    "uses_objective": False,
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
                    ],
                }
            ],
            "perturb_occlusion",
        )

        self.assertFalse(panel.objective_row.isVisible())
        self.assertEqual(set(panel.selected_method_params()), {"block_size", "stride"})
        self.assertIsInstance(panel._parameter_widgets["block_size"], QSpinBox)
        self.assertEqual(panel.selected_method_params()["block_size"], 16)

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


if __name__ == "__main__":
    unittest.main()
