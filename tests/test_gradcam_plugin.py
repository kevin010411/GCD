import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QLabel

from src.gcd.presentation.qt.plugins.gradcam import GradCamPluginPanel
from src.gcd.presentation.qt.view import MainWindowView


class GradCamPluginPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_method_combo_exists_and_uses_stable_ids(self) -> None:
        panel = GradCamPluginPanel()
        panel.method_combo.addItem("Grad-CAM", "gradcam")
        labels = {label.text() for label in panel.findChildren(QLabel)}

        self.assertIsNotNone(panel.dataset_combo)
        self.assertIsNotNone(panel.method_combo)
        self.assertIsNotNone(panel.objective_combo)
        self.assertIsNotNone(panel.layer_feature_group)
        self.assertIn("Data", labels)
        self.assertIn("Answer", labels)
        self.assertIn("Objective", labels)
        self.assertFalse(panel.layer_feature_group.isEnabled())
        self.assertEqual(panel.method_combo.count(), 1)
        self.assertEqual(panel.method_combo.itemText(0), "Grad-CAM")
        self.assertEqual(panel.method_combo.itemData(0), "gradcam")
        self.assertEqual(panel.run_button.text(), "Run")

    def test_layer_feature_controls_enable_only_after_layer_and_feature_are_available(self) -> None:
        panel = GradCamPluginPanel()
        view = MainWindowView.__new__(MainWindowView)
        view._method_options_by_id = {}
        view._gradcam_feature_size = 0
        view.layer_feature_group = panel.layer_feature_group
        view.layer_combo = panel.layer_combo
        view.method_combo = panel.method_combo
        view.feature_widget = panel.feature_widget

        MainWindowView.set_method_options(
            view,
            [{"id": "gradcam", "name": "Grad-CAM", "uses_layer_controls": True}],
            "gradcam",
        )

        self.assertFalse(panel.layer_feature_group.isEnabled())

        MainWindowView.set_layer_options(view, ["layer-a"], "layer-a")
        self.assertFalse(panel.layer_feature_group.isEnabled())

        MainWindowView.set_feature_size(view, 8)
        self.assertTrue(panel.layer_feature_group.isEnabled())

        MainWindowView.set_method_options(
            view,
            [
                {
                    "id": "saliency_map",
                    "name": "Saliency Map",
                    "uses_layer_controls": False,
                }
            ],
            "saliency_map",
        )
        self.assertFalse(panel.layer_feature_group.isEnabled())


if __name__ == "__main__":
    unittest.main()
