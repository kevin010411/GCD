from __future__ import annotations

import hashlib
import json
from pathlib import Path


def model_identity(cfg) -> str:
    """Identify architecture and weights, independently of explanation settings."""
    checkpoint = Path(str(cfg.get("ckpt", ""))).resolve()
    try:
        stat = checkpoint.stat()
        revision = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        revision = None
    settings = {"model": cfg.get("model")}
    settings.update(checkpoint=str(checkpoint).casefold(), revision=revision)
    serialized = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
