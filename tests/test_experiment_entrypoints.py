"""Guard the public CLI boundary and relocated import/config paths."""
import ast
from pathlib import Path
import unittest

from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]


class ExperimentEntrypointTests(unittest.TestCase):
    def test_only_two_public_scripts_and_package_marker(self):
        actual = {path.name for path in (ROOT / "experiments").glob("*.py")}
        self.assertEqual(actual, {"__init__.py", "benchmark.py", "evaluate_metrics.py"})
        for path in (ROOT / "experiments/src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    self.assertNotIn("__name__", ast.unparse(node.test), str(path))

    def test_relocated_configs_resolve_root_model_and_preprocessing(self):
        for path in (ROOT / "experiments/src/configs").glob("*.py"):
            cfg = Config.fromfile(str(path))
            self.assertIn("model", cfg)
            self.assertTrue(cfg.preprocessing.steps)
            self.assertEqual(len(cfg.inference.roi_size), 3)
        self.assertFalse((ROOT / "experiments/xai_benchmark.py").exists())
        self.assertFalse((ROOT / "experiments/predict.py").exists())

    def test_public_wrappers_use_source_runners(self):
        from experiments.benchmark import main as batch_main
        from experiments.evaluate_metrics import main as single_main
        from experiments.src.benchmark_runner import main as source_batch
        from experiments.src.single_case_runner import main as source_single
        self.assertIs(batch_main, source_batch)
        self.assertIs(single_main, source_single)

    def test_internal_legacy_adapters_remain_importable(self):
        from experiments.src.predict import main
        from experiments.src.runner import _xai_execution_enabled
        from experiments.src.xai.compare_scorecam_same_model import main as audit_main
        self.assertTrue(callable(main))
        self.assertTrue(callable(audit_main))
        self.assertTrue(callable(_xai_execution_enabled))


if __name__ == "__main__":
    unittest.main()
