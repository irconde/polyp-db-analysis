from __future__ import annotations

"""CLI utilities for PolypGen dataset operations."""

from pathlib import Path

from ..datasets.polypgen_build import build_polypgen_eval


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


def _run_build(args) -> None:
    track = args.track or args.root.name
    build_polypgen_eval(args.root, args.out, track=track)


__all__ = ["register"]
