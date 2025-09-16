from __future__ import annotations

"""Utilities for creating few-shot PolypGen training splits."""

from collections import defaultdict
from dataclasses import replace
from math import ceil, floor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import logging
import subprocess

import pandas as pd
import yaml

from ..manifests.core import ManifestRow, hmac_rank, pick_k
from ..manifests.io import read_csv, write_pack


logger = logging.getLogger(__name__)

# Maximum number of frames to draw from any negative sequence.
MAX_NEG_FRAMES_PER_SEQUENCE = 5


def _load_frame_metadata(
    roots: Dict[str, str],
    store_ids: Iterable[str],
    frame_ids: Iterable[str],
) -> Dict[str, Dict[str, str | None]]:
    """Load auxiliary metadata for frames referenced in ``frame_ids``.

    Parameters
    ----------
    roots:
        Mapping from ``store_id`` to dataset roots as recorded in the manifest.
    store_ids:
        Collection of ``store_id`` values referenced by the pack rows.
    frame_ids:
        Frame identifiers present in the source pack.

    Returns
    -------
    dict
        Mapping from ``frame_id`` to a dictionary with ``origin`` and
        ``sequence_id`` information when available.
    """

    metadata: Dict[str, Dict[str, str | None]] = {}
    frame_ids = set(frame_ids)
    if not frame_ids:
        return metadata

    for store_id in sorted({sid for sid in store_ids if sid}):
        root = roots.get(store_id)
        if not root:
            continue
        frames_path = Path(root) / "manifests" / "frames.parquet"
        if not frames_path.exists():
            logger.warning("frames.parquet not found for store_id %s at %s", store_id, frames_path)
            continue
        try:
            df = pd.read_parquet(
                frames_path,
                columns=["global_frame_id", "origin", "sequence_id"],
                filters=[("global_frame_id", "in", list(frame_ids))],
            )
        except Exception:
            # Fallback to loading the full table if predicate pushdown is not supported.
            try:
                df = pd.read_parquet(
                    frames_path, columns=["global_frame_id", "origin", "sequence_id"]
                )
            except Exception as exc:  # pragma: no cover - unexpected parquet issues
                logger.warning("Failed to load frame metadata from %s: %s", frames_path, exc)
                continue
            df = df[df["global_frame_id"].astype(str).isin(frame_ids)]

        if df.empty:
            continue

        for row in df.itertuples(index=False):
            frame_id = str(row.global_frame_id)
            origin = getattr(row, "origin", None)
            seq_id = getattr(row, "sequence_id", None)
            if pd.isna(origin):
                origin = None
            else:
                origin = str(origin)
            if pd.isna(seq_id):
                seq_id = None
            else:
                seq_id = str(seq_id)
            metadata[frame_id] = {"origin": origin, "sequence_id": seq_id}

    return metadata


def _select_negative_rows(
    neg_rows: List[ManifestRow],
    metadata: Dict[str, Dict[str, str | None]],
    seed: int,
    k_neg: int,
) -> Tuple[List[ManifestRow], List[str]]:
    """Select negative rows prioritising base negatives and then sequences."""

    if k_neg <= 0 or not neg_rows:
        return [], []

    base_negatives: List[ManifestRow] = []
    sequences: Dict[str, List[ManifestRow]] = defaultdict(list)
    for row in neg_rows:
        info = metadata.get(row.frame_id, {})
        seq_id = info.get("sequence_id")
        if seq_id:
            sequences[seq_id].append(row)
        elif info.get("origin") == "neg_seq":
            # Safety net: treat sequence frames without an explicit ID as their own group.
            sequences[row.frame_id].append(row)
        else:
            base_negatives.append(row)

    selected: List[ManifestRow] = []
    selected_sequences: List[str] = []

    # Stage 1: prefer base negatives with known centres.
    if base_negatives:
        take = min(k_neg, len(base_negatives))
        selected.extend(pick_k(base_negatives, seed, take, key=lambda r: r.frame_id))

    remaining = k_neg - len(selected)
    if remaining <= 0 or not sequences:
        return selected, selected_sequences

    # Stage 2: draw from negative sequences without crossing splits.
    sequence_ids = sorted(sequences.keys(), key=lambda sid: (hmac_rank(seed, sid), sid))
    for seq_id in sequence_ids:
        frames = sequences[seq_id]
        frames = sorted(frames, key=lambda r: (hmac_rank(seed, r.frame_id), r.frame_id))
        take = min(MAX_NEG_FRAMES_PER_SEQUENCE, remaining, len(frames))
        if take <= 0:
            continue
        selected.extend(frames[:take])
        selected_sequences.append(seq_id)
        remaining -= take
        if remaining <= 0:
            break

    return selected, selected_sequences


def build_polypgen_fewshot(test_pack: Path, sizes: list[int], seeds: list[int], out_dir: Path) -> None:
    """Build few-shot training/test splits from a PolypGen test pack.

    Parameters
    ----------
    test_pack:
        Path to a manifest pack containing ``test.csv`` with both positive and
        negative frames.
    sizes:
        Desired training set sizes.  For each size ``S`` and ``seed`` we select
        ``ceil(S/2)`` positive and ``floor(S/2)`` negative frames.
    seeds:
        Seeds used for deterministic selection.
    out_dir:
        Directory where the generated packs will be written.  Packs are named
        ``polypgen_fewshot_s{S}_seed{seed}``.
    """

    test_rows = read_csv(test_pack / "test.csv")

    # Load manifest for roots if available
    roots: Dict[str, str] = {}
    manifest_path = test_pack / "manifest.yaml"
    if manifest_path.exists():
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            roots = manifest.get("roots", {}) or {}
        except Exception:  # pragma: no cover - malformed manifest fallback
            roots = {}

    pos_rows = [r for r in test_rows if r.label == 1]
    neg_rows = [r for r in test_rows if r.label == 0]
    frame_metadata = _load_frame_metadata(roots, {r.store_id for r in test_rows}, {r.frame_id for r in test_rows})

    for size in sizes:
        for seed in seeds:
            k_pos = min(ceil(size / 2), len(pos_rows))
            k_neg = min(floor(size / 2), len(neg_rows))

            pos_selected = pick_k(pos_rows, seed, k_pos, key=lambda r: r.frame_id)
            neg_selected, neg_sequences = _select_negative_rows(neg_rows, frame_metadata, seed, k_neg)

            train_rows = [replace(r, split="train") for r in pos_selected + neg_selected]
            selected_ids = {r.frame_id for r in train_rows}

            neg_sequence_set = {seq_id for seq_id in neg_sequences if seq_id}
            test_rows_out: List[ManifestRow] = []
            for row in test_rows:
                if row.frame_id in selected_ids:
                    continue
                if row.label == 0 and neg_sequence_set:
                    seq_id = frame_metadata.get(row.frame_id, {}).get("sequence_id")
                    if seq_id in neg_sequence_set:
                        continue
                test_rows_out.append(replace(row, split="test"))

            pack_dir = out_dir / f"polypgen_fewshot_s{size}_seed{seed}"

            meta = {
                "name": f"polypgen_fewshot_s{size}_seed{seed}",
                "version": "0.0",
                "policy": {
                    "unit_of_split": "frame",
                    "selection": f"hmac_sha256(frame_id,{seed})",
                },
                "row_schema": {
                    "fields": [
                        {"name": "dataset", "type": "string", "desc": "constant 'polypgen_clean'"},
                        {"name": "split", "type": "string", "desc": "train|test (frame-level)"},
                        {"name": "case_id", "type": "string"},
                        {"name": "frame_id", "type": "string"},
                        {"name": "frame_path", "type": "path"},
                        {"name": "label", "type": "int"},
                    ]
                },
                "selection_keys": {
                    "train": f"hmac_sha256(frame_id,{seed})",
                    "test": "complement of train within polypgen-clean-test",
                },
                "roots": roots,
            }

            command = (
                f"polypdb polypgen fewshot --pack {test_pack} --sizes {size} "
                f"--seeds {seed} --out {out_dir}"
            )
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
            except Exception:  # pragma: no cover
                git_commit = "unknown"
            meta["generator"] = {
                "repo": "polyp-db-analysis",
                "git_commit": git_commit,
                "command": command,
                "seed": seed,
            }

            write_pack(pack_dir, {"train": train_rows, "test": test_rows_out}, meta)


__all__ = ["build_polypgen_fewshot"]
