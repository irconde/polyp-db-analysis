from __future__ import annotations

"""CLI utilities for PolypGen dataset operations."""

from pathlib import Path

from ..datasets.polypgen_build import build_polypgen_eval
from ..fewshot import build_polypgen_fewshot


def register(subparsers) -> None:
    parser = subparsers.add_parser("polypgen", help="PolypGen dataset utilities")
    sub = parser.add_subparsers(dest="polypgen_command", required=True)

    build = sub.add_parser("build", help="Build PolypGen eval manifest pack")
    build.add_argument("--root", type=Path, required=True, help="Path to PolypGen root")
    build.add_argument("--out", type=Path, required=True, help="Output directory")
    build.add_argument(
        "--track",
        type=str,
        choices=["clean", "raw"],
        default=None,
        help="Dataset track (default: infer from root name)",
    )
    build.set_defaults(func=_run_build)

    fewshot = sub.add_parser("fewshot", help="Build few-shot PolypGen packs")
    fewshot.add_argument("--pack", type=Path, required=True, help="Input eval pack")
    fewshot.add_argument(
        "--sizes", type=int, nargs="+", required=True, help="Training set sizes"
    )
    fewshot.add_argument(
        "--seeds", type=int, nargs="+", required=True, help="Selection seeds"
    )
    fewshot.add_argument("--out", type=Path, required=True, help="Output directory")
    fewshot.set_defaults(func=_run_fewshot)


def _run_build(args) -> None:
    track = args.track or args.root.name
    build_polypgen_eval(args.root, args.out, track=track)


def _run_fewshot(args) -> None:
    build_polypgen_fewshot(args.pack, args.sizes, args.seeds, args.out)


__all__ = ["register"]
