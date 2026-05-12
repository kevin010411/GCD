import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QLabel

from src.gcd.presentation.qt.plugins.perturbation import PerturbationPluginPanel


class PerturbationPluginPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_panel_exposes_dataset_method_and_run_controls(self) -> None:
        panel = PerturbationPluginPanel()
        panel.method_combo.addItem("Occlusion", "perturb_occlusion")
        labels = {label.text() for label in panel.findChildren(QLabel)}

        self.assertIsNotNone(panel.dataset_combo)
        self.assertIsNotNone(panel.class_spinbox)
        self.assertIn("Data", labels)
        self.assertIn("Answer", labels)
        self.assertEqual(panel.method_combo.itemData(0), "perturb_occlusion")
        self.assertEqual(panel.run_button.text(), "Run")
