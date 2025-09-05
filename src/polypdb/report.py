import os
import json
import hashlib
import logging
from typing import Optional

import pandas as pd
import yaml

from .merge import dataset_statistics_from_canonical


logger = logging.getLogger(__name__)


def _read_parquet_if_exists(path: str) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path) if os.path.exists(path) else None


def load_manifests(canonical_base: str) -> dict[str, Optional[pd.DataFrame]]:
    manifests = os.path.join(canonical_base, "manifests")
    return {
        "frames": _read_parquet_if_exists(os.path.join(manifests, "frames.parquet")),
        "cases": _read_parquet_if_exists(os.path.join(manifests, "cases.parquet")),
        "annotations": _read_parquet_if_exists(os.path.join(manifests, "annotations.parquet")),
    }


def write_schema_yaml(canonical_base: str, out_path: Optional[str] = None) -> str:
    manifests = os.path.join(canonical_base, "manifests")
    if out_path is None:
        out_path = os.path.join(manifests, "schema.yaml")

    schema = {
        "frames": {
            "primary_key": "global_frame_id",
            "columns": {
                "global_frame_id": "string",
                "global_case_id": "string",
                "dataset": "string",
                "source_frame_id": "string",
                "img_width": "int",
                "img_height": "int",
                "filename_ext": "string",
                "img_path": "string  # relative to canonical root",
                "modality": "string|null",
                "has_polyp": "bool",
            },
        },
        "cases": {
            "primary_key": "global_case_id",
            "columns": {
                "global_case_id": "string",
                "dataset": "string",
                "shape": "string|null  # original labels (e.g., IIa, Ip, Is)",
                "size_mm": "float|null",
                "location": "string|null",
                "n_frames": "int",
                "source_lesion_id": "string|null",
            },
        },
        "annotations": {
            "primary_key": "ann_id",
            "columns": {
                "ann_id": "string",
                "global_frame_id": "string",
                "global_case_id": "string",
                "bbox_x": "float",
                "bbox_y": "float",
                "bbox_w": "float",
                "bbox_h": "float",
                "bbox_format": "string",  # xywh
                "bbox_normalized": "bool",
            },
        },
    }

    with open(out_path, "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    logger.info("Wrote schema to %s", out_path)
    return out_path


def write_dataset_meta(canonical_base: str, out_path: Optional[str] = None) -> str:
    manifests = os.path.join(canonical_base, "manifests")
    if out_path is None:
        out_path = os.path.join(manifests, "dataset_meta.json")

    stats = dataset_statistics_from_canonical(canonical_base)
    meta = {
        "generator": "polypdb",
        "canonical_base": os.path.abspath(canonical_base),
        "stats": stats,
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote dataset metadata to %s", out_path)
    return out_path


def write_readme(canonical_base: str, out_path: Optional[str] = None) -> str:
    manifests = os.path.join(canonical_base, "manifests")
    if out_path is None:
        out_path = os.path.join(canonical_base, "README.md")

    stats = dataset_statistics_from_canonical(canonical_base)

    content = f"""
# Fused Polyp Dataset

This folder contains a fused canonical dataset with images under `images/` and
manifest tables under `manifests/`.

## Summary

- Original total frames: {stats.get('original_total_samples')}
- Total frames: {stats.get('total_samples')}
- Total cases: {stats.get('total_cases')}
- Case distribution: {stats.get('case_distribution')}
- Morphology distribution: {stats.get('morphology_distribution')}

## Directory Layout

- `images/`: canonical images. Filenames match `frames.global_frame_id + filename_ext`.
- `manifests/frames.parquet`: per-frame metadata. `img_path` is relative to this root.
- `manifests/cases.parquet` (optional): per-case metadata, `shape` values are original labels.
- `manifests/annotations.parquet`: per-bbox annotations.

## Schemas

See `manifests/schema.yaml` for a machine-readable data dictionary of all tables.

## Negatives

Negative frames are grouped into synthetic case identifiers `NEG_<DATASET>_<GROUP>`
where `<GROUP>` comes from the directory name that contained the negatives.

## Notes

- `shape` values are saved verbatim (e.g., `IIa`, `IIb`, `IIc`, `III`, `Ip`, `Is`, `Isp`).
- `img_path` fields are relative to the canonical root for portability.
""".strip() + "\n"

    with open(out_path, "w") as f:
        f.write(content)
    logger.info("Wrote README to %s", out_path)
    # Also emit schema and meta alongside
    write_schema_yaml(canonical_base, os.path.join(manifests, "schema.yaml"))
    write_dataset_meta(canonical_base, os.path.join(manifests, "dataset_meta.json"))
    return out_path


def validate_canonical(canonical_base: str) -> dict:
    dfs = load_manifests(canonical_base)
    frames = dfs["frames"]
    cases = dfs["cases"]
    ann = dfs["annotations"]
    images_root = os.path.join(canonical_base, "images")

    issues: list[str] = []
    if frames is None:
        raise FileNotFoundError("Missing frames.parquet in manifests/")

    # Key uniqueness
    if frames["global_frame_id"].duplicated().any():
        issues.append("Duplicate global_frame_id in frames")
    if cases is not None and cases["global_case_id"].duplicated().any():
        issues.append("Duplicate global_case_id in cases")
    if ann is not None and ann["ann_id"].duplicated().any():
        issues.append("Duplicate ann_id in annotations")

    # Foreign keys
    if cases is not None:
        missing_cases = set(frames["global_case_id"]) - set(cases["global_case_id"])
        if missing_cases:
            issues.append(
                f"Frames reference {len(missing_cases)} cases missing in cases.parquet"
            )
    else:
        issues.append("cases.parquet missing; skipping case validation")
    if ann is not None:
        missing_frames = set(ann["global_frame_id"]) - set(frames["global_frame_id"])
        if missing_frames:
            issues.append(
                f"Annotations reference {len(missing_frames)} frames missing in frames.parquet"
            )

    # File existence
    missing_files = []
    for rel in frames["img_path"].astype(str).tolist():
        abs_path = os.path.join(canonical_base, rel)
        if not os.path.exists(abs_path):
            missing_files.append(rel)
    if missing_files:
        issues.append(f"Missing image files: {len(missing_files)} (e.g., {missing_files[:3]})")

    ok = len(issues) == 0
    result = {"ok": ok, "issues": issues}
    logger.info("Validation %s", "passed" if ok else "found issues")
    for msg in issues:
        logger.warning("%s", msg)
    return result


def write_checksums(canonical_base: str, out_csv: Optional[str] = None) -> str:
    images_dir = os.path.join(canonical_base, "images")
    manifests = os.path.join(canonical_base, "manifests")
    if out_csv is None:
        out_csv = os.path.join(manifests, "checksums.csv")

    rows: list[dict] = []
    for root, _, files in os.walk(images_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, canonical_base)
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            rows.append({"img_path": rel, "sha256": h.hexdigest()})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote checksums for %d files to %s", len(rows), out_csv)
    return out_csv

