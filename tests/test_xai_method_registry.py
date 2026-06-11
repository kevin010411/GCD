import unittest

from src.gcd.infrastructure.methods import XaiMethod, XaiMethodRegistry


class XaiMethodRegistryTests(unittest.TestCase):
    def test_default_registry_groups_methods_by_family(self) -> None:
        registry = XaiMethodRegistry.default()

        gradient_ids = [item["id"] for item in registry.available_methods("gradient")]
        perturbation_ids = [
            item["id"] for item in registry.available_methods("perturbation")
        ]

        self.assertEqual(
            gradient_ids,
            ["gradcam", "xrescam", "gradcam_test", "saliency_map"],
        )
        self.assertEqual(perturbation_ids, ["perturb_occlusion"])

    def test_registered_methods_follow_xai_method_interface(self) -> None:
        registry = XaiMethodRegistry.default()

        for method in registry.methods_by_id.values():
            self.assertIsInstance(method, XaiMethod)

    def test_method_options_include_capabilities_and_parameter_schema(self) -> None:
        registry = XaiMethodRegistry.default()

        by_id = {
            item["id"]: item for item in registry.available_methods()
        }

        self.assertTrue(by_id["gradcam"]["uses_layer_controls"])
        self.assertTrue(by_id["saliency_map"]["uses_objective"])
        self.assertFalse(by_id["saliency_map"]["uses_layer_controls"])
        self.assertFalse(by_id["perturb_occlusion"]["uses_objective"])
        self.assertEqual(
            [item["id"] for item in by_id["perturb_occlusion"]["parameters"]],
            ["block_size", "stride"],
        )


if __name__ == "__main__":
    unittest.main()
