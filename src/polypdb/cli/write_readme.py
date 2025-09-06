"""Implementation of the ``write-readme`` sub‑command."""

from __future__ import annotations

import logging

from . import write_readme as write_readme_func

logger = logging.getLogger(__name__)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "write-readme",
        help="Generate README.md, schema.yaml, and dataset_meta.json for a canonical dataset",
    )
    parser.add_argument("canonical_dir", help="Path to the canonical dataset directory")
    parser.add_argument(
        "--readme-out", help="Optional README output path (default: <canonical>/README.md)"
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    path = write_readme_func(args.canonical_dir, args.readme_out)
    logger.info("README generated at %s", path)


__all__ = ["register"]

