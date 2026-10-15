from __future__ import annotations

"""CLI utilities for PolypGen dataset operations."""

from pathlib import Path
import yaml

from ..datasets.polypgen_build import build_polypgen_eval
from ..fewshot import (
    CANONICAL_FEWSHOT_SEED,
    CANONICAL_FEWSHOT_SIZES,
    build_polypgen_fewshot,
    build_polypgen_fewshot_canonical,
)


def register(subparsers) -> None:
    parser = subparsers.add_parser("polypgen", help="PolypGen dataset utilities")
    sub = parser.add_subparsers(dest="polypgen_command", required=True)

    build = sub.add_parser("build", help="Build PolypGen test manifest pack")
    build.add_argument("--root", type=Path, required=True, help="Path to PolypGen root")
    build.add_argument("--out", type=Path, required=True, help="Output directory")
    build.add_argument(
        "--track",
        type=str,
        choices=["clean", "raw"],
        default=None,
        help="Dataset track (default: infer from root name)",
    )
    build.add_argument(
        "--include-negative-sequences",
        action="store_true",
        help="Retain sequence-derived negatives when building the pack",
    )
    build.set_defaults(func=_run_build)

    fewshot = sub.add_parser("fewshot", help="Build few-shot PolypGen packs")
    fewshot.add_argument("--pack", type=Path, required=True, help="Input test pack")
    fewshot.add_argument(
        "--sizes", type=int, nargs="+", required=True, help="Training set sizes"
    )
    fewshot.add_argument(
        "--seeds", type=int, nargs="+", required=True, help="Selection seeds"
    )
    fewshot.add_argument("--out", type=Path, required=True, help="Output directory")
    fewshot.add_argument(
        "--dir-template",
        type=str,
        default=None,
        help="Pack directory name template with {size} and {seed} fields",
    )
    fewshot.set_defaults(func=_run_fewshot)

    fewshot_canonical = sub.add_parser(
        "fewshot-canonical", help="Build canonical few-shot PolypGen packs"
    )
    fewshot_canonical.add_argument("--pack", type=Path, required=True, help="Input test pack")
    fewshot_canonical.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Training set sizes to build (default: "
            + ", ".join(str(s) for s in CANONICAL_FEWSHOT_SIZES)
            + ")"
        ),
    )
    fewshot_canonical.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_FEWSHOT_SEED,
        help=f"Seed for canonical selection (default: {CANONICAL_FEWSHOT_SEED})",
    )
    fewshot_canonical.add_argument("--out", type=Path, required=True, help="Output directory")
    fewshot_canonical.set_defaults(func=_run_fewshot_canonical)


def _run_build(args) -> None:
    track = args.track or args.root.name
    build_polypgen_eval(
        args.root,
        args.out,
        track=track,
        include_negative_sequences=args.include_negative_sequences,
    )
    manifest = yaml.safe_load((args.out / "manifest.yaml").read_text(encoding="utf-8"))
    print(args.out)
    print(manifest.get("counts", {}))


def _run_fewshot(args) -> None:
    build_polypgen_fewshot(
        args.pack, args.sizes, args.seeds, args.out, dir_template=args.dir_template
    )
    if args.dir_template:
        for size in args.sizes:
            for seed in args.seeds:
                print(args.out / args.dir_template.format(size=size, seed=seed))
        return
    unique_seeds = set(args.seeds)
    include_seed_in_name = len(unique_seeds) > 1
    for size in args.sizes:
        for seed in args.seeds:
            suffix = f"_seed{seed}" if include_seed_in_name else ""
            pack_dir = args.out / f"polypgen_fewshot_s{size}{suffix}"
            manifest = yaml.safe_load(
                (pack_dir / "manifest.yaml").read_text(encoding="utf-8")
            )
            print(pack_dir)
            print(manifest.get("counts", {}))


def _run_fewshot_canonical(args) -> None:
    sizes = tuple(sorted(set(args.sizes))) if args.sizes else CANONICAL_FEWSHOT_SIZES
    build_polypgen_fewshot_canonical(args.pack, args.out, sizes=sizes, seed=args.seed)
    for size in sizes:
        pack_dir = args.out / f"polypgen_fewshot_s{size}"
        manifest = yaml.safe_load(
            (pack_dir / "manifest.yaml").read_text(encoding="utf-8")
        )
        print(pack_dir)
        print(manifest.get("counts", {}))


__all__ = ["register"]
