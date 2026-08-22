import os
import tempfile
import unittest

import numpy as np

from src.gcd.domain import (
    OrganIntervention,
    OrganMaskRecord,
    OrganOcclusionSpec,
)
from src.gcd.infrastructure.organ_occlusion import (
    TotalSegmentatorOrganService,
    apply_organ_occlusion,
    prediction_metrics,
    validate_mask_geometry,
)


def _record(organ_id: str, mask: np.ndarray) -> OrganMaskRecord:
    return OrganMaskRecord(
        id=organ_id,
        display_name=organ_id,
        source_labels=(organ_id,),
        mask=mask,
        affine=np.eye(4),
    )


class OrganOcclusionTests(unittest.TestCase):
    def test_fixed_hu_feather_zero_replaces_only_selected_mask(self) -> None:
        image = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
        original = image.copy()
        mask = np.zeros_like(image, dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        spec = OrganOcclusionSpec(
            interventions=(
                OrganIntervention("heart", enabled=True, mode="fixed_hu", fill_hu=-100),
            ),
            feather_mm=0,
        )

        result, modified = apply_organ_occlusion(
            image, {"heart": _record("heart", mask)}, spec
        )

        np.testing.assert_array_equal(image, original)
        self.assertTrue(np.all(result[mask] == -100))
        np.testing.assert_array_equal(result[~mask], image[~mask])
        np.testing.assert_array_equal(modified, mask)

    def test_local_mean_and_blur_return_finite_volume(self) -> None:
        image = np.zeros((7, 7, 7), dtype=np.float32)
        image[2:5, 2:5, 2:5] = 100
        first = np.zeros_like(image, dtype=bool)
        first[2:5, 2:5, 2:5] = True
        second = np.zeros_like(image, dtype=bool)
        second[1:4, 1:4, 1:4] = True
        spec = OrganOcclusionSpec(
            interventions=(
                OrganIntervention("one", enabled=True, mode="local_mean"),
                OrganIntervention("two", enabled=True, mode="gaussian_blur"),
            )
        )

        result, modified = apply_organ_occlusion(
            image,
            {"one": _record("one", first), "two": _record("two", second)},
            spec,
        )

        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_equal(modified, first | second)

    def test_empty_selection_and_invalid_hu_are_rejected(self) -> None:
        image = np.zeros((3, 3, 3), dtype=np.float32)
        mask = np.ones_like(image, dtype=bool)
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            apply_organ_occlusion(image, {"x": _record("x", mask)}, OrganOcclusionSpec(()))
        with self.assertRaisesRegex(ValueError, "between -1000 and 1000"):
            apply_organ_occlusion(
                image,
                {"x": _record("x", mask)},
                OrganOcclusionSpec((OrganIntervention("x", True, fill_hu=1001),)),
            )

    def test_geometry_validation_rejects_shape_and_affine_mismatch(self) -> None:
        image = np.zeros((4, 4, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_mask_geometry(image, np.eye(4), np.zeros((3, 3, 3)), np.eye(4))
        shifted = np.eye(4)
        shifted[0, 3] = 5
        with self.assertRaisesRegex(ValueError, "affine"):
            validate_mask_geometry(image, np.eye(4), image, shifted)

    def test_prediction_metrics_include_stability_difference_and_optional_truth(self) -> None:
        original = np.zeros((3, 3, 3), dtype=np.uint8)
        perturbed = original.copy()
        original[1, 1, 1] = 1
        perturbed[1, 1, 2] = 1
        difference, metrics = prediction_metrics(
            original,
            perturbed,
            1,
            np.ones_like(original),
            ground_truth=original,
        )
        self.assertEqual(metrics["stability_dice"], 0.0)
        self.assertIn("ground_truth_dice_delta", metrics)
        self.assertEqual(difference[1, 1, 1], 2)
        self.assertEqual(difference[1, 1, 2], 3)

    def test_cache_key_changes_with_source_content_and_version(self) -> None:
        service = TotalSegmentatorOrganService()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ct.nii.gz")
            with open(path, "wb") as handle:
                handle.write(b"one")
            first = service.cache_key(path, "1.0")
            with open(path, "wb") as handle:
                handle.write(b"two")
            second = service.cache_key(path, "1.0")
            third = service.cache_key(path, "2.0")
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

    def test_record_creation_can_disable_curated_organ_merging(self) -> None:
        labelmap = np.array([[[1, 2]]], dtype=np.uint8)
        class_map = {
            1: "lung_upper_lobe_left",
            2: "lung_lower_lobe_left",
        }

        merged = TotalSegmentatorOrganService._records_from_labelmap(
            labelmap, np.eye(4), class_map, merge_organs=True
        )
        separate = TotalSegmentatorOrganService._records_from_labelmap(
            labelmap, np.eye(4), class_map, merge_organs=False
        )

        self.assertEqual([record.id for record in merged], ["lung_left"])
        self.assertEqual(
            [record.id for record in separate],
            ["lung_upper_lobe_left", "lung_lower_lobe_left"],
        )


if __name__ == "__main__":
    unittest.main()
