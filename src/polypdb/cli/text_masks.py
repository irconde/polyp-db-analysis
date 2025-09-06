"""Implementation of the ``text-masks`` sub‑command."""

from __future__ import annotations

from ..utils import create_text_masks


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "text-masks", help="Detect text and create mask images"
    )
    parser.add_argument("src_dir", help="Directory containing input images")
    parser.add_argument("dest_dir", help="Directory to save generated mask images")
    parser.add_argument("--inpaint-dir", help="Directory to save images with text removed")
    parser.add_argument(
        "--expand",
        type=int,
        default=5,
        help="Pixels to expand each detected text mask (default: 5)",
    )
    parser.add_argument(
        "--debug-dir", help="Directory to save debug images with mask contours"
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    create_text_masks(
        args.src_dir,
        args.dest_dir,
        expand=args.expand,
        inpaint_dir=args.inpaint_dir,
        debug_dir=args.debug_dir,
    )


__all__ = ["register"]

