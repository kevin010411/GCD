"""Categorical resampling must preserve IDs and source-space geometry."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.gcd.infrastructure.xai.evaluation import export_prediction


class PredictionExportTests(unittest.TestCase):
    def test_nearest_neighbor_on_translated_finer_source_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            affine = np.diag([.5, 1., 1., 1.])
            affine[0, 3] = 10.
            source = root / 'source.nii.gz'
            image = nib.Nifti1Image(np.zeros((7, 2, 2), np.float32), affine)
            image.set_qform(affine, code=1)
            image.set_sform(affine, code=2)
            nib.save(image, source)
            model_affine = affine.copy()
            model_affine[0, 0] = 1.
            labels = np.broadcast_to(np.array([0, 1, 2, 3])[:, None, None], (4, 2, 2))
            exported = export_prediction(labels, model_affine, source, root / 'prediction.nii.gz')
            restored = nib.load(exported)
            expected = np.broadcast_to(np.array([0, 1, 1, 2, 2, 3, 3])[:, None, None], (7, 2, 2))
            np.testing.assert_array_equal(np.asarray(restored.dataobj), expected)
            np.testing.assert_allclose(restored.affine, affine)
            self.assertEqual(restored.get_data_dtype(), np.dtype('uint8'))
            self.assertEqual(restored.header.get_intent()[0], 'label')
            self.assertEqual(int(restored.header['qform_code']), 1)
            self.assertEqual(int(restored.header['sform_code']), 2)

    def test_fractional_label_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer label map'):
            export_prediction(np.full((2, 2, 2), 1.5), np.eye(4), 'unused', 'unused')
