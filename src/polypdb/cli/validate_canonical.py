"""Implementation of the ``validate-canonical`` sub‑command."""

from __future__ import annotations

from . import validate_canonical as validate_canonical_func


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-canonical", help="Validate canonical manifests and file references"
    )
    parser.add_argument("canonical_dir", help="Path to the canonical dataset directory")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    res = validate_canonical_func(args.canonical_dir)
    if not res["ok"]:
        raise SystemExit(1)


__all__ = ["register"]

