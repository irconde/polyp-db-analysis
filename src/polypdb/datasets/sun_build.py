from __future__ import annotations

"""Build manifest pack for the SUN dataset."""

from pathlib import Path
from typing import Callable, Dict, List, Any
import subprocess

from ..manifests.core import ManifestRow, pick_k
from ..manifests.io import write_pack

Index = Dict[str, Dict[str, Any]]
Indexer = Callable[[Path], Index]


def _default_indexer(root: Path) -> Index:  # pragma: no cover - real impl elsewhere
    """Placeholder indexer for the SUN dataset."""
    raise NotImplementedError("index_sun is not implemented; provide an indexer")


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

    pos_ids = sorted([cid for cid, case in index.items() if case.get("label") == 1])
    neg_ids = sorted([cid for cid, case in index.items() if case.get("label") == 0])

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
