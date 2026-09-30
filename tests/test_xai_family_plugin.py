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

    def test_scorecam_layer_and_class_remain_editable_after_result_refresh(self) -> None:
        panel = XaiFamilyPluginPanel("gradient", "Gradient XAI", "Run gradients.")
        options = [{
            "id": "scorecam",
            "name": "Score-CAM",
            "uses_layer_controls": True,
            "uses_objective": True,
            "parameters": [],
        }]
        panel.set_method_options(options, "scorecam")
        panel.set_layer_options(["encoder 1", "decoder 1"], "decoder 1", 128)
        panel.class_spinbox.setValue(2)
        panel.layer_combo.setCurrentText("encoder 1")

        self.assertTrue(panel.layer_combo.isEnabled())
        self.assertTrue(panel.class_spinbox.isEnabled())
        self.assertEqual(panel.selected_layer(), "encoder 1")
        self.assertEqual(panel.selected_class(), 2)

    def test_scorecam_defaults_to_decoder_2_without_overriding_explicit_choice(self) -> None:
        panel = XaiFamilyPluginPanel("gradient", "Gradient XAI", "Run gradients.")
        options = [
            {"id": "gradcam", "name": "Grad-CAM", "uses_layer_controls": True},
            {"id": "scorecam", "name": "Score-CAM", "uses_layer_controls": True},
        ]
        panel.set_method_options(options, "gradcam")
        panel.set_layer_options(["decoder 1", "decoder 2"], "decoder 1", 4)

        panel.method_combo.setCurrentIndex(panel.method_combo.findData("scorecam"))
        self.assertEqual(panel.selected_layer(), "decoder 2")
        self.assertFalse(panel.feature_widget.isEnabled())

        panel.layer_combo.setCurrentText("decoder 1")
        panel.set_method_options(options, "scorecam")
        panel.set_layer_options(["decoder 1", "decoder 2"], "decoder 1", 4)
        self.assertEqual(panel.selected_layer(), "decoder 1")
        self.assertEqual(panel.feature_range(), (0, 4))

        panel.set_layer_options(["decoder 1", "decoder 2"], "decoder 1", 0)
        self.assertEqual(panel.selected_layer(), "decoder 2")


if __name__ == "__main__":
    unittest.main()
