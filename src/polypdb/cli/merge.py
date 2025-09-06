"""Implementation of the ``merge`` sub‑command."""

from __future__ import annotations

from .common import load_config


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "merge", help="Merge datasets into a unified dataset"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    from . import merge_datasets

    cfg = load_config(args.config)
    merge_datasets(**cfg)


__all__ = ["register"]

