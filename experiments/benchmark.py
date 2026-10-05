"""Public batch benchmark entrypoint; implementation lives in experiments.src."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.src.benchmark_runner import main

if __name__ == "__main__":
    main()
