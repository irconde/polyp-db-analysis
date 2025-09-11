"""Command line interface for :mod:`polypdb`.

This package exposes a :func:`build_parser` function that constructs the
``argparse`` parser and registers all available sub‑commands.  Each command is
implemented in its own module within this package.  A convenience
:func:`main` function is provided for entry point usage.
"""

from __future__ import annotations

import argparse
import logging

# Re‑export merge helpers for backward compatibility and for tests that monkey
# patch these functions via ``polypdb.cli``.
from ..merge import (  # noqa: F401
    merge_datasets,
    dataset_statistics_from_canonical,
    dataset_statistics_from_parquets,
    export_manifests_to_csv,
    export_binary_classification_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the top‑level argument parser.

    The parser registers all sub‑commands by delegating to the ``register``
    function of modules in :mod:`polypdb.cli`.
    """

    parser = argparse.ArgumentParser(description="Polyp dataset processing CLI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set the logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Import sub‑command modules lazily so that ``polypdb.cli`` can be imported
    # without immediately importing all dependencies of every sub‑command.
    from . import (
        merge,
        convert,
        prepare,
        prepare_classification,
        count,
        pipeline,
        stats,
        stats_manifests,
        mask_corners,
        text_masks,
        write_readme,
        validate_canonical,
        write_checksums,
        export_csv,
        export_classification_flat,
        prepare_sun_classification,
        subset_sun_classification,
        sun,
    )

    modules = [
        merge,
        convert,
        prepare,
        prepare_classification,
        count,
        pipeline,
        stats,
        stats_manifests,
        mask_corners,
        text_masks,
        write_readme,
        validate_canonical,
        write_checksums,
        export_csv,
        export_classification_flat,
        prepare_sun_classification,
        subset_sun_classification,
        sun,
    ]

    for mod in modules:
        mod.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and dispatch to the chosen command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Each sub‑command sets ``func`` in ``set_defaults`` so that we can simply
    # call it here.
    args.func(args)


__all__ = ["build_parser", "main"]
