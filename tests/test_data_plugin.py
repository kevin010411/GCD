import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.gcd.presentation.qt.plugins.transfer_volume import TransferVolumePluginPanel


class DataPluginPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_panel_exposes_delete_control_and_signal(self) -> None:
        panel = TransferVolumePluginPanel()
        received = []
        panel.volume_list.delete_requested.connect(received.append)
        panel.volume_list.set_volumes(
            [{"id": "volume-1", "display_name": "Volume 1", "visible": True}],
            "volume-1",
        )

        panel.delete_button.click()

        self.assertEqual(panel.delete_button.text(), "Delete")
        self.assertEqual(received, ["volume-1"])


if __name__ == "__main__":
    unittest.main()
