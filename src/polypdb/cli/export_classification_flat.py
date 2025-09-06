"""Implementation of the ``export-classification-flat`` sub-command."""

from __future__ import annotations

from . import export_binary_classification_dataset


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "export-classification-flat",
        help=(
            "Export canonical dataset images into class_0/class_1 folders "
            "for binary classification"
        ),
    )
    parser.add_argument("canonical_dir", help="Path to the canonical dataset directory")
    parser.add_argument("out_dir", help="Directory to write class folders")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    export_binary_classification_dataset(args.canonical_dir, args.out_dir)


__all__ = ["register"]
