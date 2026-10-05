import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from src.gcd.infrastructure.model_catalog import load_model_catalog
from src.gcd.presentation.qt.model_selector import ModelSelectionDialog, ModelSelectorCombo


class ModelSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.options = load_model_catalog()

    def setUp(self):
        self.dialog = ModelSelectionDialog(self.options, self.options[0]["path"])
        self.addCleanup(self.dialog.close)

    def test_search_and_filters_intersect(self):
        self.dialog.search.setText("UNET MMWhS")
        self.dialog.family_filter.setCurrentIndex(self.dialog.family_filter.findData("UNet IRC"))
        self.dialog.class_filter.setCurrentIndex(self.dialog.class_filter.findData(8))
        self.assertEqual(self.dialog.table.rowCount(), 1)
        self.assertEqual(self.dialog.current_option()["id"], "unet_irc_mmwhs")
        self.dialog.accept()
        self.assertEqual(self.dialog.selected_path, self.dialog.current_option()["path"])

    def test_no_results_cannot_accept_and_clearing_restores_list(self):
        self.dialog.search.setText("no such model")
        self.assertEqual(self.dialog.table.rowCount(), 0)
        self.assertFalse(self.dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
        self.dialog.accept()
        self.assertIsNone(self.dialog.selected_path)
        self.dialog.search.clear()
        self.assertEqual(self.dialog.table.rowCount(), len(self.options))

    def test_cancel_does_not_change_active_model_and_confirm_emits_once(self):
        combo = ModelSelectorCombo()
        self.addCleanup(combo.close)
        combo.set_options(self.options)
        original = combo.currentData()
        spy = QSignalSpy(combo.currentIndexChanged)
        with patch.object(ModelSelectionDialog, "exec", return_value=QDialog.DialogCode.Rejected):
            combo.showPopup()
        self.assertEqual(combo.currentData(), original)
        self.assertEqual(len(spy), 0)

        def confirm(dialog):
            dialog.selected_path = self.options[1]["path"]
            return QDialog.DialogCode.Accepted

        with patch.object(ModelSelectionDialog, "exec", confirm):
            combo.showPopup()
        self.assertEqual(combo.currentData(), self.options[1]["path"])
        self.assertEqual(len(spy), 1)
        combo.set_options(list(reversed(self.options)))
        self.assertEqual(combo.currentData(), self.options[1]["path"])
        self.assertEqual(len(spy), 1)


if __name__ == "__main__":
    unittest.main()
