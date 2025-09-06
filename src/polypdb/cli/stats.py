"""Implementation of the ``stats`` sub‑command."""

from __future__ import annotations

from . import dataset_statistics_from_canonical


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "stats", help="Compute dataset statistics from a canonical folder"
    )
    parser.add_argument(
        "canonical_dir", help="Path to the canonical dataset directory"
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    dataset_statistics_from_canonical(args.canonical_dir)


__all__ = ["register"]

