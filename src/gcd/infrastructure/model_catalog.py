"""Explicit model menu metadata; reading the catalog never executes model configs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_CATALOG = Path(__file__).resolve().parents[3] / "config" / "model_catalog.json"


def load_model_catalog(catalog_path: str | Path = DEFAULT_MODEL_CATALOG) -> list[dict[str, Any]]:
    catalog_path = Path(catalog_path).resolve()
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read model catalog {catalog_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Unsupported model catalog version: {catalog_path}")
    entries = payload.get("models")
    if not isinstance(entries, list):
        raise ValueError("Model catalog 'models' must be a list.")
    options = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        label = f"Model catalog entry {index + 1}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be an object.")
        for field in ("id", "name", "config", "family"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"{label} requires a nonempty '{field}'.")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate model id: {entry['id']}")
        seen_ids.add(entry["id"])
        config = Path(entry["config"])
        if config.is_absolute():
            raise ValueError(f"{label} config must be relative to the catalog.")
        resolved = (catalog_path.parent / config).resolve()
        if not resolved.is_relative_to(catalog_path.parent) or resolved.suffix != ".py":
            raise ValueError(f"{label} config must be a Python file inside the catalog directory.")
        if resolved in seen_paths:
            raise ValueError(f"Duplicate model config: {config}")
        seen_paths.add(resolved)
        classes = entry.get("output_classes")
        if type(classes) is not int or classes < 1:
            raise ValueError(f"{label} output_classes must be a positive integer.")
        tags = entry.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"{label} tags must be a list of strings.")
        if not isinstance(entry.get("description", ""), str):
            raise ValueError(f"{label} description must be a string.")
        if type(entry.get("enabled", True)) is not bool:
            raise ValueError(f"{label} enabled must be a boolean.")
        if not entry.get("enabled", True):
            continue
        if not resolved.is_file():
            raise ValueError(f"Model config does not exist: {resolved}")
        options.append({**entry, "path": str(resolved), "tags": list(tags)})
    return options
