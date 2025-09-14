"""Manifest I/O utilities."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List
import csv
import random
import yaml
from datetime import datetime, timezone
from collections import Counter

from .core import ManifestRow, sha256_file

CSV_FIELDS = [
    "dataset",
    "split",
    "case_id",
    "frame_id",
    "frame_path",
    "label",
    "store_id",
    "morphology",
    "source_video",
    "frame_index",
    "width",
    "height",
    "variant",
    "synthetic_case_id",
    "source_case_id",
    "chunk_start",
    "chunk_end",
    "split_unit",
    "synthetic_case",
    "orig_frame_id",
    "perturbation_id",
    "rng_seed",
    "render_in_pipeline",
    "blur_sigma",
    "jpeg_q",
    "brightness",
    "contrast",
    "bbox_area_frac",
]


def _ensure_relative(path: str) -> None:
    p = Path(path)
    if p.is_absolute() or p.as_posix() != path:
        raise ValueError("frame_path must be relative using '/' separators")


def write_csv(rows: List[ManifestRow], path: Path) -> None:
    """Write manifest ``rows`` to ``path``.

    CSV files are UTF-8 encoded without BOM, use comma delimiters, have a
    header row, and use ``\n`` line endings.  ``frame_path`` entries are
    validated to be relative with forward slashes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            _ensure_relative(row.frame_path)
            writer.writerow({field: getattr(row, field) for field in CSV_FIELDS})


def read_csv(path: Path) -> List[ManifestRow]:
    """Read manifest rows from ``path``."""

    rows: List[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                ManifestRow(
                    dataset=raw["dataset"],
                    split=raw["split"],
                    case_id=raw["case_id"],
                    frame_id=raw["frame_id"],
                    frame_path=raw["frame_path"],
                    label=int(raw["label"]),
                    store_id=raw["store_id"],
                    morphology=raw.get("morphology", ""),
                    source_video=raw.get("source_video", ""),
                    frame_index=int(raw.get("frame_index", -1)),
                    width=int(raw.get("width", -1)),
                    height=int(raw.get("height", -1)),
                    variant=raw.get("variant", ""),
                    synthetic_case_id=raw.get("synthetic_case_id", ""),
                    source_case_id=raw.get("source_case_id", ""),
                    chunk_start=int(raw.get("chunk_start", -1)),
                    chunk_end=int(raw.get("chunk_end", -1)),
                    split_unit=raw.get("split_unit", "case"),
                    synthetic_case=raw.get("synthetic_case", "").lower()
                    in {"1", "true", "t", "yes"},
                    orig_frame_id=raw.get("orig_frame_id", ""),
                    perturbation_id=raw.get("perturbation_id", ""),
                    rng_seed=int(raw.get("rng_seed", -1)),
                    render_in_pipeline=raw.get("render_in_pipeline", "").lower()
                    in {"1", "true", "t", "yes"},
                    blur_sigma=float(raw.get("blur_sigma", -1.0)),
                    jpeg_q=int(raw.get("jpeg_q", -1)),
                    brightness=float(raw.get("brightness", -1.0)),
                    contrast=float(raw.get("contrast", -1.0)),
                    bbox_area_frac=float(raw.get("bbox_area_frac", -1.0)),
                )
            )
    return rows


def write_pack(
    out_dir: Path,
    splits: Dict[str, List[ManifestRow]],
    meta: Dict,
    selection_key_policy: Dict | None = None,
) -> None:
    """Write a manifest pack to ``out_dir``.

    ``splits`` maps split names to lists of rows.  ``meta`` supplies metadata
    stored in ``manifest.yaml``.  Optional ``selection_key_policy`` details how
    selection keys are constructed for downstream sampling.  Hashes and counts
    are computed automatically and ``generated_at`` is recorded in UTC ISO
    format.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    hashes: Dict[str, str] = {}
    counts: Dict[str, Dict[str, int]] = {}

    summary_by_split: List[Dict[str, int]] = []
    case_lengths: Dict[int, int] = {}

    for split, rows in splits.items():
        csv_path = out_dir / f"{split}.csv"
        write_csv(rows, csv_path)
        hashes[split] = sha256_file(csv_path)

        pos = sum(1 for r in rows if r.label == 1)
        neg = sum(1 for r in rows if r.label == 0)
        pos_cases_set = {r.case_id for r in rows if r.label == 1}
        neg_cases_set = {r.case_id for r in rows if r.label == 0}
        counts[split] = {
            "frames": len(rows),
            "pos": pos,
            "neg": neg,
            "pos_cases": len(pos_cases_set),
            "neg_cases": len(neg_cases_set),
        }

        flat_cases = len({r.case_id for r in rows if r.label == 1 and r.morphology == "flat"})
        polyp_cases = len({r.case_id for r in rows if r.label == 1 and r.morphology == "polypoid"})
        summary = {
            "split": split,
            "pos_cases": len(pos_cases_set),
            "neg_cases": len(neg_cases_set),
            "frames": len(rows),
            "pos_frames": pos,
            "neg_frames": neg,
            "label_counts": {"pos": pos, "neg": neg},
        }
        if flat_cases or polyp_cases:
            summary["flat_cases"] = flat_cases
            summary["polypoid_cases"] = polyp_cases
        summary_by_split.append(summary)

        case_counts = Counter(r.case_id for r in rows)
        for count in case_counts.values():
            case_lengths[count] = case_lengths.get(count, 0) + 1

    manifest = dict(meta)
    if selection_key_policy is not None:
        manifest["selection_key_policy"] = selection_key_policy
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["hashes"] = hashes
    manifest["counts"] = counts
    manifest["summary_by_split"] = summary_by_split
    if case_lengths:
        manifest["case_lengths"] = case_lengths

    manifest_path = out_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def validate_pack(
    out_dir: Path,
    require_files_exist: bool = False,
    roots: Dict[str, str] | None = None,
) -> Dict:
    """Validate a manifest pack written by :func:`write_pack` and return it.

    If ``require_files_exist`` is true, a deterministic sample of rows from each
    split is checked for file existence using ``roots`` (either provided or
    from the manifest).  SHA256 hashes recorded in ``manifest.yaml`` are
    recomputed and must match.  The parsed manifest dictionary, including any
    ``selection_key_policy`` block, is returned for further inspection.
    """

    manifest_path = out_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assertions = manifest.get("assertions", {})
    unique_key = assertions.get("unique_key", ["frame_id"])
    if isinstance(unique_key, str):
        unique_key = [unique_key]

    if roots is None:
        roots = manifest.get("roots", {})

    hashes: Dict[str, str] = manifest.get("hashes", {})
    sun_cases: Dict[str, set[str]] = {}

    for split, expected_sha in hashes.items():
        csv_path = out_dir / f"{split}.csv"
        actual_sha = sha256_file(csv_path)
        if actual_sha != expected_sha:
            raise AssertionError(f"hash mismatch for {split}")

        rows = read_csv(csv_path)

        seen_keys: set[tuple] = set()
        for row in rows:
            key = tuple(getattr(row, f) for f in unique_key)
            if key in seen_keys:
                key_str = ", ".join(f"{f}={getattr(row, f)}" for f in unique_key)
                raise AssertionError(f"duplicate key ({key_str}) in {split}")
            seen_keys.add(key)

            if row.label not in {0, 1}:
                raise AssertionError(f"invalid label {row.label} in {split}")
            if row.morphology not in {"flat", "polypoid", ""}:
                raise AssertionError(f"invalid morphology {row.morphology} in {split}")
            if row.split_unit not in {"case", "source_case"}:
                raise AssertionError(f"invalid split_unit {row.split_unit} in {split}")

        sun_cases[split] = {
            r.case_id for r in rows if r.dataset.lower() == "sun"
        }

        if require_files_exist:
            rng = random.Random(0)
            sample = rows if len(rows) <= 3 else rng.sample(rows, 3)
            for row in sample:
                root = roots.get(row.store_id)
                if root is None:
                    raise AssertionError(f"missing root for store_id {row.store_id}")
                if not (Path(root) / row.frame_path).exists():
                    raise AssertionError(
                        f"missing file {(Path(root) / row.frame_path)!s}"
                    )

    required = {"train", "val", "test"}
    if required.issubset(sun_cases):
        for a, b in combinations(sorted(required), 2):
            if not sun_cases[a].isdisjoint(sun_cases[b]):
                raise AssertionError(
                    "SUN case_id overlap between splits",
                )

    return manifest
