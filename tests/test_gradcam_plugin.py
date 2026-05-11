import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.gcd.presentation.qt.plugins.gradcam import GradCamPluginPanel


class GradCamPluginPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_method_combo_exists_and_uses_stable_ids(self) -> None:
        panel = GradCamPluginPanel()
        panel.method_combo.addItem("Grad-CAM", "gradcam")

        self.assertIsNotNone(panel.method_combo)
        self.assertEqual(panel.method_combo.count(), 1)
        self.assertEqual(panel.method_combo.itemText(0), "Grad-CAM")
        self.assertEqual(panel.method_combo.itemData(0), "gradcam")


if __name__ == "__main__":
    unittest.main()
