"""CLI for relinking negative frames to case identifiers."""
from __future__ import annotations

from ..merge import relink_negative_cases


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "relink-negative-cases",
        help="Relink negative frames and regenerate case metadata",
    )
    parser.add_argument(
        "canonical_root",
        help="Path to canonical dataset root containing manifests",
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    relink_negative_cases(args.canonical_root)


__all__ = ["register"]
