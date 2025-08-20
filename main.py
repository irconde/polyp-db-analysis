"""Command-line entry point for dataset preparation utilities."""

import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from polypdb.cli import main


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    main()
