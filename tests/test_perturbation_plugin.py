import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QProgressBar
from PyQt6.QtWidgets import QScrollArea

from src.gcd.presentation.qt.plugins.perturbation import PerturbationPluginPanel


class PerturbationPluginPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_panel_exposes_dataset_method_and_run_controls(self) -> None:
        panel = PerturbationPluginPanel()
        panel.method_combo.addItem("Occlusion", "perturb_occlusion")
        labels = {label.text() for label in panel.findChildren(QLabel)}

        self.assertIsInstance(panel.scroll_area, QScrollArea)
        self.assertTrue(panel.scroll_area.widgetResizable())
        self.assertIsNotNone(panel.dataset_combo)
        self.assertIsNotNone(panel.class_spinbox)
        self.assertIn("Data", labels)
        self.assertIn("Answer", labels)
        self.assertIn("Score", labels)
        self.assertIn("Answer Data", labels)
        self.assertEqual(panel.method_combo.itemData(0), "perturb_occlusion")
        self.assertEqual(panel.run_button.text(), "Run")
        self.assertIsInstance(panel.progress_bar, QProgressBar)
        self.assertTrue(panel.progress_bar.isHidden())

    def test_progress_bar_is_above_run_button_and_shows_counted_progress(self) -> None:
        panel = PerturbationPluginPanel()

        panel.set_progress_running(True)

        self.assertFalse(panel.progress_bar.isHidden())
        self.assertEqual(panel.progress_bar.minimum(), 0)
        self.assertEqual(panel.progress_bar.maximum(), 100)
        self.assertEqual(panel.progress_bar.format(), "0/100")
        self.assertLess(
            panel.content_layout.indexOf(panel.progress_bar),
            panel.content_layout.indexOf(panel.run_button),
        )

        panel.set_progress_value(7, 12)

        self.assertEqual(panel.progress_bar.maximum(), 12)
        self.assertEqual(panel.progress_bar.value(), 7)
        self.assertEqual(panel.progress_bar.format(), "7/12")

        panel.set_progress_running(False)

        self.assertTrue(panel.progress_bar.isHidden())
        self.assertEqual(panel.progress_bar.maximum(), 100)

    def test_answer_data_options_can_reference_loaded_data_volume(self) -> None:
        panel = PerturbationPluginPanel()

        panel.set_answer_data_options(
            [{"id": "dataset-1:base", "name": "patient-label"}],
            "dataset-1:base",
        )

        self.assertEqual(panel.selected_answer_data(), "dataset-1:base")

    def test_preview_controls_expose_checkbox_and_pause_state(self) -> None:
        panel = PerturbationPluginPanel()

        self.assertFalse(panel.preview_enabled())
        panel.preview_checkbox.setChecked(True)
        panel.set_preview_running(True)

        self.assertTrue(panel.preview_enabled())
        self.assertFalse(panel.preview_pause_button.isHidden())

        panel.preview_pause_button.setChecked(True)

        self.assertTrue(panel.preview_paused())
        self.assertEqual(panel.preview_pause_button.text(), "Resume")

        panel.set_preview_running(False)

        self.assertTrue(panel.preview_pause_button.isHidden())
        self.assertEqual(panel.preview_pause_button.text(), "Pause")
