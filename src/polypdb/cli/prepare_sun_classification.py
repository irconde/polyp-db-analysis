"""Implementation of the ``prepare-sun-classification`` sub-command."""

from __future__ import annotations

from ..sun_classification import prepare_sun_classification


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare-sun-classification",
        help="Prepare SUN dataset for binary classification",
    )
    parser.add_argument("--src", required=True, help="Path to canonical SUN dataset")
    parser.add_argument("--dest", required=True, help="Output directory")
    parser.add_argument(
        "--stratify-morphology",
        action="store_true",
        help="Stratify positive cases by Paris morphology",
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    prepare_sun_classification(
        args.src, args.dest, stratify_morphology=args.stratify_morphology
    )


__all__ = ["register"]
