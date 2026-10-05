import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.gcd.infrastructure.model_catalog import load_model_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_shipped_catalog_matches_config_class_counts(self):
        options = load_model_catalog()
        self.assertTrue(options)
        for option in options:
            tree = ast.parse(Path(option["path"]).read_text(encoding="utf-8"))
            model = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "model"
                                 for target in node.targets))
            fields = {kw.arg: ast.literal_eval(kw.value) for kw in model.keywords}
            self.assertEqual(option["output_classes"], fields.get("out_channels", fields.get("num_classes")))

    def test_explicit_order_disabled_entries_and_no_config_execution(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b", "unlisted"):
                (root / f"{name}.py").write_text("raise RuntimeError('must not execute')", encoding="utf-8")
            entries = [self.entry("b"), self.entry("a"), {**self.entry("disabled"), "enabled": False}]
            catalog = self.write(root, entries)
            result = load_model_catalog(catalog)
            self.assertEqual([item["id"] for item in result], ["b", "a"])
            self.assertTrue(all(Path(item["path"]).is_absolute() for item in result))

    def test_invalid_catalog_reports_actionable_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").touch()
            cases = [
                ([self.entry("a"), self.entry("a")], "Duplicate model id"),
                ([self.entry("missing")], "does not exist"),
                ([{**self.entry("a"), "config": "../outside.py"}], "inside"),
                ([{**self.entry("a"), "output_classes": True}], "positive integer"),
                ([{**self.entry("a"), "tags": "mmwhs"}], "list of strings"),
                ([{**self.entry("a"), "enabled": "false"}], "boolean"),
            ]
            for entries, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        load_model_catalog(self.write(root, entries))
            catalog = self.write(root, [])
            catalog.write_text('{"version": 2, "models": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                load_model_catalog(catalog)

    def test_workflow_catalog_loading_is_independent_of_working_directory(self):
        from src.gcd.application.services import WorkflowService

        original = Path.cwd()
        expected = load_model_catalog()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                options = WorkflowService(object()).list_model_configs()
                self.assertEqual(options, expected)
                self.assertTrue(all(Path(option["path"]).is_file() for option in options))
            finally:
                os.chdir(original)

    @staticmethod
    def entry(name):
        return {"id": name, "name": name, "config": f"{name}.py", "family": "UNet", "output_classes": 4}

    @staticmethod
    def write(root, entries):
        catalog = root / "catalog.json"
        catalog.write_text(json.dumps({"version": 1, "models": entries}), encoding="utf-8")
        return catalog


if __name__ == "__main__":
    unittest.main()
