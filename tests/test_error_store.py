import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.gcd.infrastructure.error_store import ErrorStore


class ErrorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_save_json_uses_expected_suffix(self):
        store = ErrorStore(str(self.tempdir))
        path = store.save_json({"error_type": "model_load_error"}, suffix="model_load_error")
        self.assertTrue(path.name.endswith("_model_load_error.json"))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["error_type"], "model_load_error")


if __name__ == "__main__":
    unittest.main()
