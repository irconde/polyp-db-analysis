from __future__ import annotations

from pathlib import Path

from ..merge import rebuild_parquet_manifests


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "rebuild-parquets",
        help="Rebuild frames.parquet and cases.parquet from metadata.xlsx",
    )
    parser.add_argument("metadata", help="Path to metadata.xlsx")
    parser.add_argument(
        "canonical_dir", help="Canonical dataset directory containing images"
    )
    parser.add_argument(
        "--images-dir",
        help="Directory with images (default: canonical_dir/images)",
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    images_dir = args.images_dir or Path(args.canonical_dir) / "images"
    rebuild_parquet_manifests(args.metadata, args.canonical_dir, images_dir)


__all__ = ["register"]
