import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.gcd.presentation.qt.plugins.registry import DEFAULT_PLUGIN_DEFINITIONS


class PluginRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_registry_defines_existing_plugins_with_stable_ids(self) -> None:
        plugin_ids = [definition.plugin_id for definition in DEFAULT_PLUGIN_DEFINITIONS]
        button_labels = [
            definition.button_label for definition in DEFAULT_PLUGIN_DEFINITIONS
        ]

        self.assertEqual(
            plugin_ids,
            ["data", "camera", "gradcam", "perturbation", "roi"],
        )
        self.assertEqual(button_labels[0], "Data")

    def test_registry_factories_build_panels(self) -> None:
        panels = [
            definition.panel_factory(None)
            for definition in DEFAULT_PLUGIN_DEFINITIONS
        ]

        self.assertTrue(all(panel is not None for panel in panels))


if __name__ == "__main__":
    unittest.main()
