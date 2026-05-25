__all__ = [
    "MODEL",
    "build_model",
    "timer",
]


def __getattr__(name: str):
    if name in {"MODEL", "build_model"}:
        from .register import MODEL, build_model

        return {"MODEL": MODEL, "build_model": build_model}[name]
    if name == "timer":
        from .utils import timer

        return timer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
