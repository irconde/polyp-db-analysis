"""Implementation of the ``mask-corners`` sub‑command."""

from __future__ import annotations

from ..utils import crop_and_mask_corner_artifacts


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "mask-corners", help="Mask corner artifacts then crop images"
    )
    parser.add_argument("src_dir", help="Directory containing input images")
    parser.add_argument("dest_dir", help="Directory to save processed images")
    parser.add_argument(
        "--padding-color",
        default="15,15,15",
        help="RGB padding colour to strip after masking (default: 15,15,15)",
    )
    parser.add_argument(
        "--padding-threshold",
        type=int,
        default=20,
        help="Colour deviation allowed when removing padding (default: 20)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Colour deviation threshold for corner artefact detection",
    )
    parser.add_argument(
        "--max-auto-fraction",
        type=float,
        default=0.15,
        help=(
            "Maximum fraction of the image dimension that auto detection may mask "
            "when removing corner artefacts (default: 0.15)"
        ),
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    padding_color = tuple(int(c) for c in args.padding_color.split(","))
    crop_and_mask_corner_artifacts(
        args.src_dir,
        args.dest_dir,
        padding_color=padding_color,
        padding_threshold=args.padding_threshold,
        threshold=args.threshold,
        max_auto_fraction=args.max_auto_fraction,
    )


__all__ = ["register"]

