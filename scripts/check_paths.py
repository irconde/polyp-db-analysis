#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from polypdb.manifests.core import ManifestRow
from polypdb.utils.roots import load_roots, resolve_path


def parse_row(data: dict[str, str]) -> ManifestRow:
    """Construct a :class:`ManifestRow` from CSV data."""

    return ManifestRow(
        dataset=data["dataset"],
        split=data["split"],
        case_id=data["case_id"],
        frame_id=data["frame_id"],
        frame_path=data["frame_path"],
        label=int(data["label"]),
        store_id=data["store_id"],
        morphology=data.get("morphology", ""),
        source_video=data.get("source_video", ""),
        frame_index=int(data.get("frame_index", -1)),
        width=int(data.get("width", -1)),
        height=int(data.get("height", -1)),
        variant=data.get("variant", ""),
        synthetic_case_id=data.get("synthetic_case_id", ""),
        source_case_id=data.get("source_case_id", ""),
        chunk_start=int(data.get("chunk_start", -1)),
        chunk_end=int(data.get("chunk_end", -1)),
        split_unit=data.get("split_unit", "case"),
        synthetic_case=data.get("synthetic_case", "").lower() in {"1", "true", "t", "yes"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check file paths from a manifest CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Manifest CSV file")
    parser.add_argument(
        "--roots",
        type=Path,
        help="JSON mapping of store_id to filesystem root",
    )
    parser.add_argument(
        "--default-root",
        type=Path,
        help="Fallback root for unknown store_id values",
    )
    args = parser.parse_args()

    manifest_yaml = args.csv.parent / "manifest.yaml"
    roots = load_roots(
        args.roots,
        manifest_yaml if manifest_yaml.exists() else None,
        args.default_root,
    )

    total = 0
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for data in reader:
            row = parse_row(data)
            path = resolve_path(row, roots)
            if not path.exists():
                print(f"missing {path}")
                print(f"checked {total} files")
                sys.exit(1)
            total += 1

    print(f"checked {total} files")


if __name__ == "__main__":
    main()
