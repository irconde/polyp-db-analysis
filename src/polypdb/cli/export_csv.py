"""Implementation of the ``export-csv`` sub‑command."""

from __future__ import annotations

import logging

from . import export_manifests_to_csv

logger = logging.getLogger(__name__)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "export-csv",
        help=(
            "Export manifest parquet files (frames/lesions/annotations) as CSV from "
            "a canonical dataset directory"
        ),
    )
    parser.add_argument("canonical_dir", help="Path to the canonical dataset directory")
    parser.add_argument("out_dir", help="Directory to write CSV files")
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=["frames", "cases", "annotations"],
        help="Subset of tables to export (default: all)",
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    written = export_manifests_to_csv(args.canonical_dir, args.out_dir, args.tables)
    if written:
        for name, path in written.items():
            logger.info("Exported %s -> %s", name, path)
    else:
        logger.warning("No tables were exported.")


__all__ = ["register"]

