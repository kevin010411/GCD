from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ModelLoadStateDictError(RuntimeError):
    def __init__(self, message: str, *, error_file: str) -> None:
        super().__init__(message)
        self.error_file = error_file
        self.skip_error_store = True


@dataclass(frozen=True)
class ModelRuntime:
    model: Any
    device: Any
    checkpoint_path: str


class ModelRuntimeLoader:
    def __init__(
        self,
        *,
        build_model: Callable[[object], Any],
        error_store: object | None = None,
    ) -> None:
        self._build_model = build_model
        self.error_store = error_store

    def load(self, cfg) -> ModelRuntime:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(cfg.model).to(device)
        checkpoint_path = str(cfg.ckpt)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        pth = torch.load(checkpoint_path, map_location="cpu")
        state_dict = pth["state_dict"].copy() if "state_dict" in pth else pth.copy()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            error_path = self._save_model_load_error(
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                missing=missing,
                unexpected=unexpected,
            )
            raise ModelLoadStateDictError(
                (
                    "Model load failed because checkpoint keys do not match the model."
                    + (
                        f" Error details saved to: {error_path}"
                        if error_path
                        else ""
                    )
                ),
                error_file=str(error_path) if error_path else "",
            )
        model.eval()
        return ModelRuntime(
            model=model,
            device=device,
            checkpoint_path=checkpoint_path,
        )

    def _save_model_load_error(
        self,
        *,
        cfg,
        checkpoint_path: str,
        missing,
        unexpected,
    ) -> str | None:
        if self.error_store is None:
            return None
        return self.error_store.save_json(
            {
                "error_type": "model_load_error",
                "config_path": getattr(cfg, "filename", None),
                "checkpoint_path": checkpoint_path,
                "missing_count": len(missing),
                "missing_keys": list(missing),
                "unexpected_count": len(unexpected),
                "unexpected_keys": list(unexpected),
            },
            suffix="model_load_error",
        )
