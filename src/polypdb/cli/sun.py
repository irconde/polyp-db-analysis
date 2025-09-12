"""CLI utilities for SUN dataset operations."""

from pathlib import Path
import yaml

from ..datasets.sun_build import build_sun_full, build_sun_morphology
from ..datasets.sun_corrupt import build_sun_corrupt
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

    morph = sub.add_parser(
        "morph", help="Build SUN morphology-stratified manifest pack"
    )
    morph.add_argument("--root", type=Path, required=True, help="Path to SUN dataset root")
    morph.add_argument("--out", type=Path, required=True, help="Output directory")
    morph.add_argument("--seed", type=int, required=True, help="Selection seed")
    morph.add_argument(
        "--frames-per-case",
        type=int,
        default=33,
        help="Frames to retain per case (default: 33)",
    )
    morph.set_defaults(func=_run_morph)

    subsets = sub.add_parser("subsets", help="Create SUN train subset packs")
    subsets.add_argument("--pack", type=Path, required=True, help="Path to sun_full pack")
    subsets.add_argument("--percents", type=int, nargs="+", required=True, help="Percents of cases per class to retain")
    subsets.add_argument("--seeds", type=int, nargs="+", required=True, help="Selection seeds")
    subsets.add_argument("--out", type=Path, required=True, help="Output directory")
    subsets.set_defaults(func=_run_subsets)

    corrupt = sub.add_parser("corrupt", help="Create corrupted SUN pack")
    corrupt.add_argument("--pack", type=Path, required=True, help="Path to sun_full pack")
    corrupt.add_argument("--out", type=Path, required=True, help="Output directory")
    corrupt.add_argument(
        "--variant",
        type=str,
        default="corrupt",
        help="Variant label for corrupted frames (default: corrupt)",
    )
    corrupt.set_defaults(func=_run_corrupt)



def _run_build(args) -> None:
    build_sun_full(
        args.root,
        args.out,
        args.seed,
        frames_per_case=args.frames_per_case,
    )
    manifest = yaml.safe_load((args.out / "manifest.yaml").read_text(encoding="utf-8"))
    print(args.out)
    print(manifest.get("counts", {}))



def _run_subsets(args) -> None:
    build_sun_train_subsets(args.pack, args.percents, args.seeds, args.out)
    for percent in args.percents:
        for seed in args.seeds:
            pack_dir = args.out / f"sun_p{percent:02d}_seed{seed}"
            manifest = yaml.safe_load(
                (pack_dir / "manifest.yaml").read_text(encoding="utf-8")
            )
            print(pack_dir)
            print(manifest.get("counts", {}))


def _run_corrupt(args) -> None:
    build_sun_corrupt(args.pack, args.out, variant=args.variant)
    manifest = yaml.safe_load((args.out / "manifest.yaml").read_text(encoding="utf-8"))
    print(args.out)
    print(manifest.get("counts", {}))


def _run_morph(args) -> None:
    build_sun_morphology(
        args.root,
        args.out,
        args.seed,
        frames_per_case=args.frames_per_case,
    )
    manifest = yaml.safe_load((args.out / "manifest.yaml").read_text(encoding="utf-8"))
    print(args.out)
    print(manifest.get("counts", {}))

__all__ = ["register"]
