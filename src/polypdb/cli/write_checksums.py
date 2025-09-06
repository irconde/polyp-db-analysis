"""Implementation of the ``write-checksums`` sub‑command."""

from __future__ import annotations

import logging

from . import write_checksums as write_checksums_func

logger = logging.getLogger(__name__)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "write-checksums",
        help="Write sha256 checksums for images/ to manifests/checksums.csv",
    )
    parser.add_argument("canonical_dir", help="Path to the canonical dataset directory")
    parser.add_argument(
        "--out", help="Optional checksums CSV path (default: manifests/checksums.csv)"
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    out = write_checksums_func(args.canonical_dir, args.out)
    logger.info("Checksums written to %s", out)


__all__ = ["register"]

