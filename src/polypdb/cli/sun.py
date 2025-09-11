"""CLI utilities for SUN dataset operations."""

from pathlib import Path

from ..datasets.sun_build import build_sun_full
from ..subsets.sun_subsets import build_sun_train_subsets

def register(subparsers) -> None:
    parser = subparsers.add_parser("sun", help="SUN dataset utilities")
    sub = parser.add_subparsers(dest="sun_command", required=True)

    build = sub.add_parser("build", help="Build SUN full manifest pack")
    build.add_argument("--root", type=Path, required=True, help="Path to SUN dataset root")
    build.add_argument("--out", type=Path, required=True, help="Output directory")
    build.add_argument("--seed", type=int, required=True, help="Selection seed")
    build.add_argument(
        "--frames-per-case",
        type=int,
        default=33,
        help="Frames to retain per case (default: 33)",
    )
    build.set_defaults(func=_run_build)
    subsets = sub.add_parser("subsets", help="Create SUN train subset packs")
    subsets.add_argument("--pack", type=Path, required=True, help="Path to sun_full pack")
    subsets.add_argument("--percents", type=int, nargs="+", required=True, help="Percents of cases per class to retain")
    subsets.add_argument("--seeds", type=int, nargs="+", required=True, help="Selection seeds")
    subsets.add_argument("--out", type=Path, required=True, help="Output directory")
    subsets.set_defaults(func=_run_subsets)



def _run_build(args) -> None:
    build_sun_full(
        args.root,
        args.out,
        args.seed,
        frames_per_case=args.frames_per_case,
    )



def _run_subsets(args) -> None:
    build_sun_train_subsets(args.pack, args.percents, args.seeds, args.out)

__all__ = ["register"]
