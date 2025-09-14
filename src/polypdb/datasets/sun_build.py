from __future__ import annotations

"""Build manifest pack for the SUN dataset."""

from pathlib import Path
from typing import Callable, Dict, List, Any
import subprocess
import logging
from itertools import combinations
from collections import Counter

from ..manifests.core import ManifestRow, pick_k, hmac_rank
from ..manifests.io import write_pack

logger = logging.getLogger(__name__)

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


def _split_targets(total: int) -> Dict[str, int]:
    """Return exact train/val/test counts for ``total`` items."""

    n_train = round(total * 0.7)
    n_val = round(total * 0.15)
    n_test = total - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test}


def _draw_ids(case_ids: List[str], targets: Dict[str, int]) -> tuple[List[str], List[str], List[str]]:
    """Draw exact numbers of ``case_ids`` according to ``targets``."""

    total_requested = sum(targets.values())
    if total_requested > len(case_ids):  # pragma: no cover - safety check
        raise ValueError("split targets exceed available case ids")
    n_train = targets.get("train", 0)
    n_val = targets.get("val", 0)
    n_test = targets.get("test", 0)
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
    if not no_case_leak:
        logger.debug("Case leakage across splits detected: %s", case_sets)
        if verify:
            raise AssertionError("case leakage across splits")

    split_counts: Dict[str, Dict[str, int]] = {}
    for split, rows in splits.items():
        pos_cases = len({r.case_id for r in rows if r.label == 1})
        neg_cases = len({r.case_id for r in rows if r.label == 0})
        split_counts[split] = {
            "pos_cases": pos_cases,
            "neg_cases": neg_cases,
            "frames": len(rows),
        }
    total_pos = sum(c["pos_cases"] for c in split_counts.values())
    total_neg = sum(c["neg_cases"] for c in split_counts.values())
    pos_targets = _split_targets(total_pos)
    neg_targets = _split_targets(total_neg)

    expected: Dict[str, Dict[str, int]] = {}
    for split in split_counts.keys():
        exp_pos = pos_targets.get(split, 0)
        exp_neg = neg_targets.get(split, 0)
        expected[split] = {
            "pos_cases": exp_pos,
            "neg_cases": exp_neg,
            "frames": frames_per_case * (exp_pos + exp_neg),
        }

    mismatches: Dict[str, Dict[str, tuple[int, int]]] = {}
    for split, counts in split_counts.items():
        split_mismatches = {
            k: (counts[k], expected[split][k])
            for k in counts
            if counts[k] != expected[split][k]
        }
        if split_mismatches:
            mismatches[split] = split_mismatches

    balanced = all(
        counts["pos_cases"] == counts["neg_cases"]
        for counts in split_counts.values()
    )
    if mismatches:
        logger.debug("Split target mismatches detected: %s", mismatches)
    if not balanced:
        logger.debug("Split counts not balanced: %s", split_counts)
    if verify and (mismatches or not balanced):
        raise AssertionError("split counts do not meet targets")

    case_frame_counts = Counter()
    for rows in splits.values():
        for r in rows:
            case_frame_counts[r.case_id] += 1
    exceptions = sorted(
        cid for cid, cnt in case_frame_counts.items() if cnt != frames_per_case
    )
    if exceptions:
        logger.debug("Cases with unexpected frame counts: %s", exceptions)
        if verify:
            raise AssertionError("cases do not have expected number of frames")

    assertions: Dict[str, Any] = {
        "no_case_leakage_across_splits": no_case_leak,
        "split_targets_met": split_counts,
        "all_cases_have_33_frames": True if not exceptions else exceptions,
    }
    if mismatches:
        assertions["split_target_mismatches"] = mismatches
    return assertions


def _allocate_negative_cases(
    neg_cases: Dict[str, Dict[str, Any]],
    seed: int,
    frames_per_case: int,
    *,
    targets: Dict[str, int] | None = None,
    verify: bool = True,
) -> tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, List[str]],
    List[Dict[str, Any]],
    Dict[str, int],
    Dict[str, int],
]:
    """Chunk and allocate negative cases to splits without leakage.

    Returns the virtual negative cases mapping, the split membership of their
    identifiers, ``neg_sources`` metadata entries, counts of original negative
    cases per split, and the (possibly adjusted) split targets.
    """

    if targets is None:
        targets = {"train": 70, "val": 15, "test": 15}

    # Partition each negative case into contiguous chunks of ``frames_per_case``
    # frames and rank the cases/chunks deterministically.
    chunks_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for cid, case in neg_cases.items():
        frames = case["frames"]
        num_chunks = len(frames) // frames_per_case
        if num_chunks == 0:
            continue
        case_chunks: List[Dict[str, Any]] = []
        for idx in range(num_chunks):
            start = idx * frames_per_case
            end = start + frames_per_case
            start_idx = int(frames[start].get("frame_index", start))
            end_idx = int(frames[end - 1].get("frame_index", end - 1))
            case_chunks.append(
                {
                    "chunk_idx": idx,
                    "frames": frames[start:end],
                    "start": start_idx,
                    "end": end_idx,
                }
            )
        chunks_by_case[cid] = case_chunks

    total_available = sum(len(v) for v in chunks_by_case.values())
    target_total = sum(targets.values())
    if total_available < target_total:
        factor = total_available / target_total if target_total else 0
        train = int(round(targets["train"] * factor))
        val = int(round(targets["val"] * factor))
        targets = {"train": train, "val": val, "test": total_available - train - val}

    caps = {s: max(1, int(targets[s] * 0.4)) for s in targets}
    remaining = dict(targets)

    ranked_cases = sorted(
        chunks_by_case.keys(), key=lambda c: (hmac_rank(seed, c), c)
    )

    virtual_cases: Dict[str, Dict[str, Any]] = {}
    split_ids: Dict[str, List[str]] = {s: [] for s in targets}
    split_sources: Dict[str, set[str]] = {s: set() for s in targets}
    per_source_counts: Dict[str, Dict[str, int]] = {s: {} for s in targets}

    for cid in ranked_cases:
        chunks = chunks_by_case[cid]
        ranked_chunks = sorted(
            chunks,
            key=lambda ch: (hmac_rank(seed, f"{cid}/{ch['chunk_idx']}") , ch["chunk_idx"]),
        )
        for split in ("train", "val", "test"):
            if remaining[split] <= 0:
                continue
            allowed = min(len(ranked_chunks), remaining[split], caps[split])
            if allowed <= 0:
                continue
            for ch in ranked_chunks[:allowed]:
                vid = f"{cid}_v{ch['chunk_idx']:03d}"
                virtual_cases[vid] = {
                    "label": 0,
                    "frames": ch["frames"],
                    "synthetic_case_id": vid,
                    "source_case_id": cid,
                    "chunk_start": ch["start"],
                    "chunk_end": ch["end"],
                }
                split_ids[split].append(vid)
            remaining[split] -= allowed
            split_sources[split].add(cid)
            per_source_counts[split][cid] = allowed
            break
        if all(r == 0 for r in remaining.values()):
            break

    if verify and any(r > 0 for r in remaining.values()):
        raise AssertionError("negative targets not met")

    for split, counts in per_source_counts.items():
        if verify and any(v > caps[split] for v in counts.values()):
            raise AssertionError("per-source cap exceeded")
        if verify and len(split_sources[split]) < 3:
            raise AssertionError("fewer than 3 negative sources assigned")

    seen: set[str] = set()
    for srcs in split_sources.values():
        if verify and not seen.isdisjoint(srcs):
            raise AssertionError("negative case leakage across splits")
        seen.update(srcs)

    neg_sources_meta: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for cid in split_sources[split]:
            vids = [vid for vid in split_ids[split] if vid.startswith(f"{cid}_v")]
            spans = [
                [virtual_cases[vid]["chunk_start"], virtual_cases[vid]["chunk_end"]]
                for vid in vids
            ]
            neg_sources_meta.append(
                {
                    "source_case_id": cid,
                    "split": split,
                    "synthetic_case_ids": vids,
                    "chunk_spans": spans,
                }
            )
    neg_sources_meta.sort(key=lambda x: (x["source_case_id"], x["split"]))

    source_counts = {s: len(split_sources[s]) for s in targets}

    return virtual_cases, split_ids, neg_sources_meta, source_counts, targets


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

    # Separate positive and negative cases from the index.  Negatives are
    # further split into virtual cases deterministically with per‑source
    # exclusivity across splits.

    pos_cases = {cid: case for cid, case in index.items() if case.get("label") == 1}
    neg_cases = {cid: case for cid, case in index.items() if case.get("label") == 0}

    (
        virtual_neg_cases,
        neg_split_ids,
        neg_sources_meta,
        neg_source_counts,
        neg_targets,
    ) = _allocate_negative_cases(
        neg_cases, seed, frames_per_case, verify=verify
    )

    index = {**pos_cases, **virtual_neg_cases}

    logger.debug(
        "Positive cases available: %d, negative cases available: %d",
        len(pos_cases),
        len(virtual_neg_cases),
    )

    pos_ids = sorted(pos_cases.keys())
    pos_selected = pick_k(pos_ids, seed, min(100, len(pos_ids)), key=lambda x: x)

    logger.debug("Selected positive cases: %d", len(pos_selected))

    pos_train, pos_val, pos_test = _split_cases(pos_selected)

    split_case_ids = {
        "train": pos_train + neg_split_ids["train"],
        "val": pos_val + neg_split_ids["val"],
        "test": pos_test + neg_split_ids["test"],
    }

    split_membership = {
        "train": {"pos_cases": pos_train, "neg_cases": neg_split_ids["train"]},
        "val": {"pos_cases": pos_val, "neg_cases": neg_split_ids["val"]},
        "test": {"pos_cases": pos_test, "neg_cases": neg_split_ids["test"]},
    }

    logger.debug("split_case_ids: %s", split_case_ids)
    logger.debug("split_membership: %s", split_membership)

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
                        split_unit="case",
                        synthetic_case=bool(case.get("synthetic_case_id")),
                    )
                )
    assertions = _compute_assertions(splits, frames_per_case, verify=verify)
    neg_targets_actual = {s: len(ids) for s, ids in neg_split_ids.items()}
    assertions["negative_case_targets"] = {
        "expected": neg_targets,
        "actual": neg_targets_actual,
    }
    assertions["no_negative_case_leakage"] = True
    assertions["neg_sources_per_split"] = neg_source_counts

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
            "negative_split_unit": "case",
            "frames_per_case": frames_per_case,
            "balance": "cases_pos=70,cases_neg=70",
            "selection": "hmac_sha256(case_id,seed)",
        },
        "frame_sampling": {
            "selection": "hmac_sha256(frame_path,salt)",
            "path_normalization": "posix,no-leading-slash,single-slash",
            "path_is_relative_to": "roots.sun",
            "salt_id": f"seed-{seed}",
            "tie_break": "frame_path",
            "frames_per_case": frames_per_case,
            "resampled_with_replacement": False,
        },
        "roots": {"sun": str(sun_root)},
        "assertions": assertions,
        "splits": split_membership,
    }

    if neg_sources_meta:
        meta["neg_sources"] = neg_sources_meta

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

    logger.debug(
        "Initial counts - flat positives: %d, polypoid positives: %d, negatives: %d",
        len(flat_ids),
        len(polypoid_ids),
        len(neg_cases),
    )

    (
        virtual_neg_cases,
        neg_split_ids,
        neg_sources_meta,
        neg_source_counts,
        neg_targets,
    ) = _allocate_negative_cases(
        neg_cases, seed, frames_per_case, verify=verify
    )

    index = {**pos_cases, **virtual_neg_cases}

    logger.debug(
        "Positive cases available - flat: %d, polypoid: %d; negative cases available: %d",
        len(flat_ids),
        len(polypoid_ids),
        len(virtual_neg_cases),
    )

    flat_selected = pick_k(sorted(flat_ids), seed, len(flat_ids), key=lambda x: x)
    polyp_selected = pick_k(sorted(polypoid_ids), seed, len(polypoid_ids), key=lambda x: x)

    logger.debug(
        "Selected cases - flat: %d, polypoid: %d", len(flat_selected), len(polyp_selected)
    )

    flat_targets = _split_targets(len(flat_selected))
    polyp_targets = _split_targets(len(polyp_selected))

    logger.debug(
        "Split targets - flat: %s, polypoid: %s, negatives: %s",
        flat_targets,
        polyp_targets,
        neg_targets,
    )

    flat_train, flat_val, flat_test = _draw_ids(flat_selected, flat_targets)
    polyp_train, polyp_val, polyp_test = _draw_ids(polyp_selected, polyp_targets)

    split_case_ids = {
        "train": flat_train + polyp_train + neg_split_ids["train"],
        "val": flat_val + polyp_val + neg_split_ids["val"],
        "test": flat_test + polyp_test + neg_split_ids["test"],
    }

    split_membership = {
        "train": {
            "pos_flat_cases": flat_train,
            "pos_polypoid_cases": polyp_train,
            "neg_cases": neg_split_ids["train"],
        },
        "val": {
            "pos_flat_cases": flat_val,
            "pos_polypoid_cases": polyp_val,
            "neg_cases": neg_split_ids["val"],
        },
        "test": {
            "pos_flat_cases": flat_test,
            "pos_polypoid_cases": polyp_test,
            "neg_cases": neg_split_ids["test"],
        },
    }

    logger.debug("split_case_ids: %s", split_case_ids)
    logger.debug("split_membership: %s", split_membership)

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
                        split_unit="case",
                        synthetic_case=bool(case.get("synthetic_case_id")),
                    )
                )
    assertions = _compute_assertions(splits, frames_per_case, verify=verify)
    neg_targets_actual = {s: len(ids) for s, ids in neg_split_ids.items()}
    assertions["negative_case_targets"] = {
        "expected": neg_targets,
        "actual": neg_targets_actual,
    }
    assertions["no_negative_case_leakage"] = True
    assertions["neg_sources_per_split"] = neg_source_counts

    neg_sources = neg_sources_meta

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
        "train": {"flat": flat_targets["train"], "polypoid": polyp_targets["train"]},
        "val": {"flat": flat_targets["val"], "polypoid": polyp_targets["val"]},
        "test": {"flat": flat_targets["test"], "polypoid": polyp_targets["test"]},
    }
    logger.debug("Morphology targets actual: %s", morph_targets_actual)
    logger.debug("Expected morphology targets: %s", expected_morph_targets)
    if morph_targets_actual != expected_morph_targets:
        logger.debug(
            "Morphology targets mismatch: expected %s, actual %s",
            expected_morph_targets,
            morph_targets_actual,
        )
        if verify:
            raise AssertionError("morphology targets not met")
    assertions["morphology_targets"] = {
        "expected": expected_morph_targets,
        "actual": morph_targets_actual,
    }

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
            "negative_split_unit": "case",
            "frames_per_case": frames_per_case,
            "balance": "cases_pos=70,cases_neg=70,stratify=morphology",
            "selection": "hmac_sha256(case_id,seed)",
        },
        "frame_sampling": {
            "selection": "hmac_sha256(frame_path,salt)",
            "path_normalization": "posix,no-leading-slash,single-slash",
            "path_is_relative_to": "roots.sun",
            "salt_id": f"seed-{seed}",
            "tie_break": "frame_path",
            "frames_per_case": frames_per_case,
            "resampled_with_replacement": False,
        },
        "roots": {"sun": str(sun_root)},
        "assertions": assertions,
        "morphology_map": morph_map,
        "morphology_counts": morph_counts,
        "splits": split_membership,
    }

    if neg_sources:
        meta["neg_sources"] = neg_sources

    write_pack(out_dir, splits, meta)


__all__ = ["build_sun_full", "build_sun_morphology"]
