from __future__ import annotations

"""Build manifest pack for the SUN dataset."""

from pathlib import Path
from typing import Callable, Dict, List, Any
import subprocess

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


def build_sun_full(
    sun_root: Path,
    out_dir: Path,
    seed: int,
    frames_per_case: int = 33,
    *,
    indexer: Indexer | None = None,
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
            virtual_id = f"{cid}_v{i:03d}"
            virtual_neg_cases[virtual_id] = {
                "label": 0,
                "frames": frames[start:end],
            }

    # Replace the original negatives with the virtual cases.
    index = {**pos_cases, **virtual_neg_cases}

    pos_ids = sorted(pos_cases.keys())
    neg_ids = sorted(virtual_neg_cases.keys())

    pos_selected = pick_k(pos_ids, seed, min(100, len(pos_ids)), key=lambda x: x)
    neg_selected = pick_k(neg_ids, seed, min(100, len(neg_ids)), key=lambda x: x)

    pos_train, pos_val, pos_test = _split_cases(pos_selected)
    neg_train, neg_val, neg_test = _split_cases(neg_selected)

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
                    )
                )

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
            "frames_per_case": frames_per_case,
            "balance": "cases_pos=70,cases_neg=70",
            "selection": "hmac_sha256(path,seed)",
        },
        "roots": {"sun": str(sun_root)},
    }

    write_pack(out_dir, splits, meta)


__all__ = ["build_sun_full"]
