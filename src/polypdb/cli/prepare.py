"""Implementation of the ``prepare`` sub‑command."""

from __future__ import annotations

from ..ultralytics import prepare_dataset


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare", help="Prepare dataset for Ultralytics YOLO"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    prepare_dataset(args.config)


__all__ = ["register"]

