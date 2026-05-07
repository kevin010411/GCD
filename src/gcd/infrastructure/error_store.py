from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import traceback


class ErrorStore:
    def __init__(self, root: str = "output/error") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, error: Exception, *, context: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", context).strip("_").lower() or "error"
        path = self.root / f"{timestamp}_{slug}.log"
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        path.write_text(trace, encoding="utf-8")
        print(f"error log saved to: {path}")
        return path

    def save_json(self, payload: dict, *, suffix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", suffix).strip("_").lower() or "error"
        path = self.root / f"{timestamp}_{slug}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"error log saved to: {path}")
        return path
