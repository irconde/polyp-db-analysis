"""Implementation of the ``subset-sun-classification`` sub-command."""

from __future__ import annotations

from ..sun_classification import create_sun_classification_subset


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "subset-sun-classification",
        help="Create a training subset of a SUN classification dataset",
    )
    parser.add_argument("--src", required=True, help="Path to prepared SUN classification dataset")
    parser.add_argument("--dest", required=True, help="Output directory for the subset")
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of training cases to sample (0-1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    create_sun_classification_subset(args.src, args.dest, args.fraction, args.seed)


__all__ = ["register"]
