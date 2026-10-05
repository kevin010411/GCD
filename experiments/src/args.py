from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any


class ConfigOptionsAction(argparse.Action):
    """Parse KEY=VALUE overrides without loading MMEngine for ``--help``."""

    def __call__(self, parser, namespace, values: list, option_string=None) -> None:
        options: dict[str, Any] = {}
        for item in values:
            if "=" not in item:
                raise argparse.ArgumentError(self, f"expected KEY=VALUE, got: {item}")
            key, raw_value = item.split("=", 1)
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                value = {"true": True, "false": False, "none": None}.get(
                    raw_value.lower(), raw_value
                )
            options[key] = value
        setattr(namespace, self.dest, options)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GCD segmentation on one NIfTI volume or a dataset directory."
    )
    parser.add_argument(
        "input", type=Path, help="Input .nii/.nii.gz volume or dataset directory"
    )
    parser.add_argument("--config", type=Path, required=True, help="MMEngine config")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Result directory containing NIfTI, JSON, CSV, and plots",
    )
    parser.add_argument(
        "--ground-truth", type=Path, help="Optional label NIfTI for Dice/IoU"
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Metrics JSON filename inside the result directory (default: metrics.json)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=ConfigOptionsAction,
        help="Override config values, e.g. inference.device=cpu",
    )
    return parser.parse_args(argv)
