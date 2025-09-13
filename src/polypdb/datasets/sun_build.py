from __future__ import annotations

"""Build manifest pack for the SUN dataset."""

from pathlib import Path
from typing import Callable, Dict, List, Any
import subprocess
from itertools import combinations
from collections import Counter

from ..manifests.core import ManifestRow, pick_k
from ..manifests.io import write_pack

Index = Dict[str, Dict[str, Any]]
Indexer = Callable[[Path], Index]


def _shape_to_morphology(value: Any) -> str:
    """Map raw shape or Paris codes to manifest morphology labels.

    Returns "flat", "polypoid", or an empty string when the value cannot be
    classified.
    """

    if value is None:
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""

    flat = {"iia", "iib", "iic", "iii", "flat", "depressed"}
    polypoid = {
        "is",
        "isp",
        "ip",
        "sessile",
        "pedunculated",
        "semipedunculated",
        "polypoid",
    }

    if s in flat or "flat" in s or "depress" in s:
        return "flat"
    if s in polypoid:
        return "polypoid"
    return ""


def _default_indexer(root: Path) -> Index:
    """Return a case → frame mapping for a canonical SUN dataset.

    The canonical layout produced by :mod:`polypdb.preprocess.sun` stores
    frame‑level metadata in ``manifests/frames.parquet``.  This function parses
    that file and returns the structure expected by :func:`build_sun_full`:

    ``{case_id: {"label": int, "frames": [{...}]}}``

    Only a minimal subset of fields is extracted so that the resulting manifest
    contains the frame path (relative to ``root``), label, and size information.
    Missing width/height values are filled by probing the image files when
    possible; otherwise ``-1`` is used.
    """

    import pandas as pd
    from PIL import Image

    frames_path = root / "manifests" / "frames.parquet"
    if not frames_path.exists():
        raise FileNotFoundError(f"Cannot locate frames metadata: {frames_path}")

    df = pd.read_parquet(frames_path)

    cases_path = root / "manifests" / "cases.parquet"
    case_morph: Dict[str, str] = {}
    if cases_path.exists():
        cases_df = pd.read_parquet(cases_path)
        if "global_case_id" in cases_df.columns:
            shape_col = None
            if "shape" in cases_df.columns:
                shape_col = "shape"
            elif "paris_class" in cases_df.columns:
                shape_col = "paris_class"
            if shape_col:
                for row in cases_df[["global_case_id", shape_col]].itertuples(index=False):
                    cid = str(getattr(row, "global_case_id"))
                    case_morph[cid] = _shape_to_morphology(getattr(row, shape_col))

    index: Index = {}
    for case_id, group in df.groupby("global_case_id"):
        frames: List[Dict[str, Any]] = []
        label = 1 if bool(group["has_polyp"].iloc[0]) else 0

        for row in group.itertuples(index=False):
            frame_path = getattr(row, "img_path", None)
            if frame_path is None:
                # fall back to global_frame_id with jpeg extension
                frame_path = f"images/{row.global_frame_id}.jpg"

            width = getattr(row, "WIDTH", getattr(row, "width", -1))
            height = getattr(row, "HEIGHT", getattr(row, "height", -1))
            if (width is None or width <= 0 or height is None or height <= 0):
                try:
                    with Image.open(root / frame_path) as im:
                        width, height = im.size
                except Exception:  # pragma: no cover - best effort
                    width, height = -1, -1

            frame_index = -1
            gfid = getattr(row, "global_frame_id", "")
            if gfid:
                try:
                    frame_index = int(str(gfid).split(".")[-1])
                except ValueError:
                    frame_index = -1

            morphology = getattr(row, "morphology", getattr(row, "paris_class", ""))
            if morphology is None or (hasattr(morphology, "__float__") and pd.isna(morphology)):
                morphology = ""
            morphology = _shape_to_morphology(morphology) or case_morph.get(str(case_id), "")

            source_video = getattr(row, "source_video", "")

            frames.append(
                {
                    "frame_path": Path(frame_path).as_posix(),
                    "frame_index": int(frame_index),
                    "width": int(width) if width is not None else -1,
                    "height": int(height) if height is not None else -1,
                    "morphology": morphology,
                    "source_video": source_video,
                }
            )

        index[str(case_id)] = {"label": label, "frames": frames}

    return index


def _split_cases(case_ids: List[str]) -> tuple[List[str], List[str], List[str]]:
    train = case_ids[:70]
    val = case_ids[70:85]
    test = case_ids[85:100]
    return train, val, test


def _split_by_ratio(case_ids: List[str]) -> tuple[List[str], List[str], List[str]]:
    """Split ``case_ids`` into train/val/test using 70/15/15 ratios."""

    n = len(case_ids)
    n_train = round(n * 0.7)
    n_val = round(n * 0.15)
    n_test = n - n_train - n_val
    train = case_ids[:n_train]
    val = case_ids[n_train : n_train + n_val]
    test = case_ids[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def _compute_assertions(
    splits: Dict[str, List[ManifestRow]],
    frames_per_case: int,
    *,
    verify: bool = True,
) -> Dict[str, Any]:
    """Compute split integrity assertions.

    Parameters
    ----------
    splits:
        Mapping of split name to manifest rows.
    frames_per_case:
        Expected number of frames per case.
    verify:
        When ``True`` raise ``AssertionError`` if invariants are not met.
    """

    case_sets = {s: {r.case_id for r in rows} for s, rows in splits.items()}
    no_case_leak = all(
        case_sets[a].isdisjoint(case_sets[b])
        for a, b in combinations(case_sets, 2)
    )
    if verify and not no_case_leak:
        raise AssertionError("case leakage across splits")

    neg_sources = {
        s: {r.source_case_id for r in rows if r.label == 0}
        for s, rows in splits.items()
    }
    no_neg_leak = all(
        neg_sources[a].isdisjoint(neg_sources[b])
        for a, b in combinations(neg_sources, 2)
    )
    if verify and not no_neg_leak:
        raise AssertionError("negative source leakage across splits")

    split_counts: Dict[str, Dict[str, int]] = {}
    for split, rows in splits.items():
        pos_cases = len({r.case_id for r in rows if r.label == 1})
        neg_cases = len({r.case_id for r in rows if r.label == 0})
        split_counts[split] = {
            "pos_cases": pos_cases,
            "neg_cases": neg_cases,
            "frames": len(rows),
        }

    expected = {
        "train": {"pos_cases": 70, "neg_cases": 70, "frames": frames_per_case * 140},
        "val": {"pos_cases": 15, "neg_cases": 15, "frames": frames_per_case * 30},
        "test": {"pos_cases": 15, "neg_cases": 15, "frames": frames_per_case * 30},
    }
    targets_met = split_counts == expected
    balanced = all(
        counts["pos_cases"] == counts["neg_cases"]
        for counts in split_counts.values()
    )
    if verify and not (targets_met and balanced):
        raise AssertionError("split counts do not meet targets")

    case_frame_counts = Counter()
    for rows in splits.values():
        for r in rows:
            case_frame_counts[r.case_id] += 1
    exceptions = sorted(
        cid for cid, cnt in case_frame_counts.items() if cnt != frames_per_case
    )
    if verify and exceptions:
        raise AssertionError("cases do not have expected number of frames")

    assertions: Dict[str, Any] = {
        "no_case_leakage_across_splits": no_case_leak,
        "no_negative_source_leakage": no_neg_leak,
        "split_targets_met": split_counts,
        "all_cases_have_33_frames": True if not exceptions else exceptions,
    }
    return assertions


def build_sun_full(
    sun_root: Path,
    out_dir: Path,
    seed: int,
    frames_per_case: int = 33,
    *,
    indexer: Indexer | None = None,
    verify: bool = True,
) -> None:
    """Build a full SUN manifest pack.

    Parameters
    ----------
    sun_root:
        Path to the SUN dataset root.
    out_dir:
        Directory where the manifest pack will be written.
    seed:
        Seed used for deterministic selection of cases and frames.
    frames_per_case:
        Number of frames to retain per case (default: 33).
    indexer:
        Optional custom indexer returning the SUN case mapping.  If omitted a
        placeholder ``index_sun`` implementation is used which must be
        overridden in tests or by consumers.
    """

    if indexer is None:
        indexer = _default_indexer

    index = indexer(sun_root)

    # Separate positive and negative cases from the index so that we can
    # transform the negatives into synthetic "virtual" cases.  The SUN dataset
    # contains only 13 negative videos which are very long.  For training we
    # need 100 negative cases with similar size to the positives, so each
    # negative video is partitioned into non‑overlapping contiguous chunks of
    # ``frames_per_case`` frames.  Each chunk becomes its own case with a
    # synthetic identifier derived from the source case id.

    pos_cases = {cid: case for cid, case in index.items() if case.get("label") == 1}
    neg_cases = {cid: case for cid, case in index.items() if case.get("label") == 0}

    virtual_neg_cases: Dict[str, Dict[str, Any]] = {}
    for cid, case in neg_cases.items():
        frames = case["frames"]
        chunks = len(frames) // frames_per_case
        for i in range(chunks):
            start = i * frames_per_case
            end = start + frames_per_case
            start_idx = int(frames[start].get("frame_index", start))
            end_idx = int(frames[end - 1].get("frame_index", end - 1))
            virtual_id = f"{cid}_v{i:03d}"
            virtual_neg_cases[virtual_id] = {
                "label": 0,
                "frames": frames[start:end],
                "synthetic_case_id": virtual_id,
                "source_case_id": cid,
                "chunk_start": start_idx,
                "chunk_end": end_idx,
            }

    # Replace the original negatives with the virtual cases.
    index = {**pos_cases, **virtual_neg_cases}

    pos_ids = sorted(pos_cases.keys())
    neg_ids = sorted(virtual_neg_cases.keys())

    pos_selected = pick_k(pos_ids, seed, min(100, len(pos_ids)), key=lambda x: x)
    neg_selected = pick_k(neg_ids, seed, min(100, len(neg_ids)), key=lambda x: x)

    pos_train, pos_val, pos_test = _split_cases(pos_selected)

    neg_groups: Dict[str, List[str]] = {}
    neg_source_order: List[str] = []
    for vid in neg_selected:
        src = virtual_neg_cases[vid]["source_case_id"]
        if src not in neg_groups:
            neg_groups[src] = []
            neg_source_order.append(src)
        neg_groups[src].append(vid)

    neg_train_src, neg_val_src, neg_test_src = _split_by_ratio(neg_source_order)
    neg_train = [vid for src in neg_train_src for vid in neg_groups[src]]
    neg_val = [vid for src in neg_val_src for vid in neg_groups[src]]
    neg_test = [vid for src in neg_test_src for vid in neg_groups[src]]

    split_case_ids = {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test,
    }

    splits: Dict[str, List[ManifestRow]] = {"train": [], "val": [], "test": []}

    for split, case_ids in split_case_ids.items():
        for case_id in case_ids:
            case = index[case_id]
            frames = pick_k(
                case["frames"], seed, frames_per_case, key=lambda r: r["frame_path"]
            )
            for frame in frames:
                frame_id = f"{case_id}/{frame['frame_path']}"
                splits[split].append(
                    ManifestRow(
                        dataset="SUN",
                        split=split,
                        case_id=case_id,
                        frame_id=frame_id,
                        frame_path=frame["frame_path"],
                        label=case["label"],
                        store_id="sun",
                        morphology=frame.get("morphology", ""),
                        source_video=frame.get("source_video", ""),
                        frame_index=int(frame.get("frame_index", -1)),
                        width=int(frame.get("width", -1)),
                        height=int(frame.get("height", -1)),
                        synthetic_case_id=case.get("synthetic_case_id", ""),
                        source_case_id=case.get("source_case_id", case_id),
                        chunk_start=int(case.get("chunk_start", -1)),
                        chunk_end=int(case.get("chunk_end", -1)),
                        split_unit="source_case" if case["label"] == 0 else "case",
                    )
                )
    assertions = _compute_assertions(splits, frames_per_case, verify=verify)

    command = (
        f"polypdb sun build --root {sun_root} --out {out_dir} "
        f"--seed {seed} --frames-per-case {frames_per_case}"
    )
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:  # pragma: no cover - fallback when git unavailable
        git_commit = "unknown"

    meta = {
        "name": "sun-full",
        "version": "0.0",
        "generator": {
            "repo": "polyp-db-analysis",
            "git_commit": git_commit,
            "command": command,
            "seed": seed,
        },
        "policy": {
            "unit_of_split": "case",
            "negative_split_unit": "source_case",
            "frames_per_case": frames_per_case,
            "balance": "cases_pos=70,cases_neg=70",
            "selection": "hmac_sha256(path,seed)",
        },
        "roots": {"sun": str(sun_root)},
        "assertions": assertions,
    }

    write_pack(out_dir, splits, meta)


def build_sun_morphology(
    sun_root: Path,
    out_dir: Path,
    seed: int,
    frames_per_case: int = 33,
    *,
    indexer: Indexer | None = None,
    verify: bool = True,
) -> None:
    """Build a SUN manifest pack stratified by morphology."""

    if indexer is None:
        indexer = _default_indexer

    index = indexer(sun_root)

    pos_cases: Dict[str, Dict[str, Any]] = {}
    neg_cases: Dict[str, Dict[str, Any]] = {}
    flat_ids: List[str] = []
    polypoid_ids: List[str] = []

    for cid, case in index.items():
        if case.get("label") == 1:
            morph = ""
            for f in case["frames"]:
                m = f.get("morphology", "")
                if m:
                    morph = m
                    break
            case = dict(case)
            case["morphology"] = morph
            pos_cases[cid] = case
            if morph == "flat":
                flat_ids.append(cid)
            else:
                polypoid_ids.append(cid)
        else:
            neg_cases[cid] = case

    virtual_neg_cases: Dict[str, Dict[str, Any]] = {}
    for cid, case in neg_cases.items():
        frames = case["frames"]
        chunks = len(frames) // frames_per_case
        for i in range(chunks):
            start = i * frames_per_case
            end = start + frames_per_case
            start_idx = int(frames[start].get("frame_index", start))
            end_idx = int(frames[end - 1].get("frame_index", end - 1))
            vid = f"{cid}_v{i:03d}"
            virtual_neg_cases[vid] = {
                "label": 0,
                "frames": frames[start:end],
                "synthetic_case_id": vid,
                "source_case_id": cid,
                "chunk_start": start_idx,
                "chunk_end": end_idx,
            }

    index = {**pos_cases, **virtual_neg_cases}

    flat_selected = pick_k(sorted(flat_ids), seed, len(flat_ids), key=lambda x: x)
    polyp_selected = pick_k(sorted(polypoid_ids), seed, len(polypoid_ids), key=lambda x: x)
    neg_ids = sorted(virtual_neg_cases.keys())
    neg_selected = pick_k(neg_ids, seed, min(100, len(neg_ids)), key=lambda x: x)

    neg_groups: Dict[str, List[str]] = {}
    neg_source_order: List[str] = []
    for vid in neg_selected:
        src = virtual_neg_cases[vid]["source_case_id"]
        if src not in neg_groups:
            neg_groups[src] = []
            neg_source_order.append(src)
        neg_groups[src].append(vid)

    flat_train, flat_val, flat_test = _split_by_ratio(flat_selected)
    polyp_train, polyp_val, polyp_test = _split_by_ratio(polyp_selected)
    neg_train_src, neg_val_src, neg_test_src = _split_by_ratio(neg_source_order)
    neg_train = [vid for src in neg_train_src for vid in neg_groups[src]]
    neg_val = [vid for src in neg_val_src for vid in neg_groups[src]]
    neg_test = [vid for src in neg_test_src for vid in neg_groups[src]]

    split_case_ids = {
        "train": flat_train + polyp_train + neg_train,
        "val": flat_val + polyp_val + neg_val,
        "test": flat_test + polyp_test + neg_test,
    }

    splits: Dict[str, List[ManifestRow]] = {"train": [], "val": [], "test": []}

    for split, case_ids in split_case_ids.items():
        for case_id in case_ids:
            case = index[case_id]
            frames = pick_k(
                case["frames"], seed, frames_per_case, key=lambda r: r["frame_path"]
            )
            for frame in frames:
                frame_id = f"{case_id}/{frame['frame_path']}"
                splits[split].append(
                    ManifestRow(
                        dataset="SUN",
                        split=split,
                        case_id=case_id,
                        frame_id=frame_id,
                        frame_path=frame["frame_path"],
                        label=case["label"],
                        store_id="sun",
                        morphology=frame.get("morphology", ""),
                        source_video=frame.get("source_video", ""),
                        frame_index=int(frame.get("frame_index", -1)),
                        width=int(frame.get("width", -1)),
                        height=int(frame.get("height", -1)),
                        synthetic_case_id=case.get("synthetic_case_id", ""),
                        source_case_id=case.get("source_case_id", case_id),
                        chunk_start=int(case.get("chunk_start", -1)),
                        chunk_end=int(case.get("chunk_end", -1)),
                        split_unit="source_case" if case["label"] == 0 else "case",
                    )
                )
    assertions = _compute_assertions(splits, frames_per_case, verify=verify)

    morph_map: Dict[str, str] = {}
    split_morph_counts: Dict[str, Dict[str, int]] = {}
    overall_counts = {
        "flat_cases": 0,
        "polypoid_cases": 0,
        "flat_frames": 0,
        "polypoid_frames": 0,
    }
    morph_targets_actual: Dict[str, Dict[str, int]] = {}

    for split, rows in splits.items():
        flat_case_set: set[str] = set()
        polyp_case_set: set[str] = set()
        flat_frames = 0
        polyp_frames = 0
        for r in rows:
            if r.label == 1 and r.morphology:
                morph_map.setdefault(r.case_id, r.morphology)
                if r.morphology == "flat":
                    flat_case_set.add(r.case_id)
                    flat_frames += 1
                elif r.morphology == "polypoid":
                    polyp_case_set.add(r.case_id)
                    polyp_frames += 1
        split_morph_counts[split] = {
            "flat_cases": len(flat_case_set),
            "polypoid_cases": len(polyp_case_set),
            "flat_frames": flat_frames,
            "polypoid_frames": polyp_frames,
        }
        morph_targets_actual[split] = {
            "flat": len(flat_case_set),
            "polypoid": len(polyp_case_set),
        }
        overall_counts["flat_cases"] += len(flat_case_set)
        overall_counts["polypoid_cases"] += len(polyp_case_set)
        overall_counts["flat_frames"] += flat_frames
        overall_counts["polypoid_frames"] += polyp_frames

    morph_counts = {"overall": overall_counts, **split_morph_counts}

    expected_morph_targets = {
        "train": {"flat": len(flat_train), "polypoid": len(polyp_train)},
        "val": {"flat": len(flat_val), "polypoid": len(polyp_val)},
        "test": {"flat": len(flat_test), "polypoid": len(polyp_test)},
    }
    if verify and morph_targets_actual != expected_morph_targets:
        raise AssertionError("morphology targets not met")
    assertions["morphology_targets_met"] = morph_targets_actual

    command = (
        f"polypdb sun morph --root {sun_root} --out {out_dir} "
        f"--seed {seed} --frames-per-case {frames_per_case}"
    )
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:  # pragma: no cover - fallback when git unavailable
        git_commit = "unknown"

    meta = {
        "name": "sun-morphology",
        "version": "0.0",
        "generator": {
            "repo": "polyp-db-analysis",
            "git_commit": git_commit,
            "command": command,
            "seed": seed,
        },
        "policy": {
            "unit_of_split": "case",
            "negative_split_unit": "source_case",
            "frames_per_case": frames_per_case,
            "balance": "cases_pos=70,cases_neg=70,stratify=morphology",
            "selection": "hmac_sha256(path,seed)",
        },
        "roots": {"sun": str(sun_root)},
        "assertions": assertions,
        "morphology_map": morph_map,
        "morphology_counts": morph_counts,
    }

    write_pack(out_dir, splits, meta)


__all__ = ["build_sun_full", "build_sun_morphology"]
