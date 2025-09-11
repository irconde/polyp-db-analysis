"""Manifest I/O utilities."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List
import csv
import random
import yaml
from datetime import datetime, timezone

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
    "severity",
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
                    severity=raw.get("severity", ""),
                )
            )
    return rows


def write_pack(out_dir: Path, splits: Dict[str, List[ManifestRow]], meta: Dict) -> None:
    """Write a manifest pack to ``out_dir``.

    ``splits`` maps split names to lists of rows.  ``meta`` supplies metadata
    stored in ``manifest.yaml``.  Hashes and counts are computed automatically
    and ``generated_at`` is recorded in UTC ISO format.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    hashes: Dict[str, str] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for split, rows in splits.items():
        csv_path = out_dir / f"{split}.csv"
        write_csv(rows, csv_path)
        hashes[split] = sha256_file(csv_path)

        pos = sum(1 for r in rows if r.label == 1)
        neg = sum(1 for r in rows if r.label == 0)
        pos_cases = len({r.case_id for r in rows if r.label == 1})
        neg_cases = len({r.case_id for r in rows if r.label == 0})
        counts[split] = {
            "frames": len(rows),
            "pos": pos,
            "neg": neg,
            "pos_cases": pos_cases,
            "neg_cases": neg_cases,
        }

    manifest = dict(meta)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["hashes"] = hashes
    manifest["counts"] = counts

    manifest_path = out_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def validate_pack(
    out_dir: Path,
    require_files_exist: bool = False,
    roots: Dict[str, str] | None = None,
) -> None:
    """Validate a manifest pack written by :func:`write_pack`.

    If ``require_files_exist`` is true, a deterministic sample of rows from each
    split is checked for file existence using ``roots`` (either provided or
    from the manifest).  SHA256 hashes recorded in ``manifest.yaml`` are
    recomputed and must match.
    """

    manifest_path = out_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

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

        seen_ids: set[str] = set()
        for row in rows:
            if row.frame_id in seen_ids:
                raise AssertionError(f"duplicate frame_id {row.frame_id} in {split}")
            seen_ids.add(row.frame_id)

            if row.label not in {0, 1}:
                raise AssertionError(f"invalid label {row.label} in {split}")
            if row.morphology not in {"flat", "polypoid", ""}:
                raise AssertionError(f"invalid morphology {row.morphology} in {split}")

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
                    "SUN case_id overlap between splits"
                )
