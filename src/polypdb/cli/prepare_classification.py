"""Implementation of the ``prepare-classification`` sub‑command."""

from __future__ import annotations

from ..ultralytics import prepare_classification_dataset


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare-classification",
        help="Prepare dataset for image classification",
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    prepare_classification_dataset(args.config)


__all__ = ["register"]

