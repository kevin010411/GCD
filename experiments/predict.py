from __future__ import annotations

from .args import parse_args
from .runner import run


def main(argv: list[str] | None = None) -> int:
    """Assemble the CLI parser and prediction runner."""
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
