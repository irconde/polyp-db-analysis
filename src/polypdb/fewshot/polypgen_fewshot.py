from __future__ import annotations

"""Utilities for creating few-shot PolypGen training splits."""

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, cast
import logging
import subprocess

import pandas as pd
import yaml

from ..manifests.core import ManifestRow, hmac_rank, pick_k
from ..manifests.io import read_csv, write_pack


logger = logging.getLogger(__name__)

# Maximum number of frames to draw from any negative sequence.
MAX_NEG_FRAMES_PER_SEQUENCE = 5

VAL_FRACTION_RULE = {
    "size_lt_100": 0.10,
    "size_100_to_199": 0.15,
    "size_gte_200": 0.20,
}


@dataclass
class NegativeSelectionResult:
    """Outcome of the negative frame selection procedure."""

    rows: List[ManifestRow]
    sequence_ids: List[str]
    sequence_frames: Dict[str, set[str]]
    base_selected_count: int


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
) -> NegativeSelectionResult:
    """Select negative rows using the pool-first capped sequence policy."""

    if k_neg <= 0 or not neg_rows:
        return NegativeSelectionResult([], [], {}, 0)

    base_negatives: List[ManifestRow] = []
    sequences: Dict[str, List[ManifestRow]] = defaultdict(list)
    sequence_frames: Dict[str, set[str]] = {}

    for row in neg_rows:
        info = metadata.get(row.frame_id, {})
        origin = (info.get("origin") or "").lower()
        seq_id_raw = info.get("sequence_id")
        seq_id = str(seq_id_raw) if seq_id_raw else ""
        if seq_id:
            sequences[seq_id].append(row)
            sequence_frames.setdefault(seq_id, set()).add(row.frame_id)
            continue
        if origin == "neg_seq":
            seq_key = f"neg_seq::{row.frame_id}"
            sequences[seq_key].append(row)
            sequence_frames.setdefault(seq_key, set()).add(row.frame_id)
            continue
        base_negatives.append(row)

    selected: List[ManifestRow] = []
    base_selected: List[ManifestRow] = []
    selected_sequences: List[str] = []

    if base_negatives:
        take = min(k_neg, len(base_negatives))
        base_selected = pick_k(base_negatives, seed, take, key=lambda r: r.frame_id)
        selected.extend(base_selected)

    remaining = k_neg - len(selected)
    if remaining > 0 and sequences:
        sequence_ids = sorted(
            sequences.keys(), key=lambda sid: (hmac_rank(seed, sid), sid)
        )
        for seq_id in sequence_ids:
            if remaining <= 0:
                break
            frames = sequences[seq_id]
            frames = sorted(
                frames, key=lambda r: (hmac_rank(seed, r.frame_id), r.frame_id)
            )
            take = min(MAX_NEG_FRAMES_PER_SEQUENCE, remaining, len(frames))
            if take <= 0:
                continue
            selected.extend(frames[:take])
            selected_sequences.append(seq_id)
            remaining -= take

    sequence_frames_selected: Dict[str, set[str]] = {
        seq_id: set(sequence_frames.get(seq_id, set())) for seq_id in selected_sequences
    }

    return NegativeSelectionResult(
        selected,
        selected_sequences,
        sequence_frames_selected,
        len(base_selected),
    )


def build_polypgen_fewshot(test_pack: Path, sizes: list[int], seeds: list[int], out_dir: Path) -> None:
    """Build few-shot training/test splits from a PolypGen test pack.

    Parameters
    ----------
    test_pack:
        Path to a manifest pack containing ``test.csv`` with both positive and
        negative frames.
    sizes:
        Desired training set sizes.  For each size ``S`` and ``seed`` we target
        ``S/2`` positive and ``S/2`` negative frames, relaxing the negative
        quota when the available pool is exhausted.
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
    manifest_name = None
    if manifest_path.exists():
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            roots = manifest.get("roots", {}) or {}
            manifest_name = manifest.get("name")
        except Exception:  # pragma: no cover - malformed manifest fallback
            roots = {}
            manifest_name = None

    parent_pack_label = manifest_name or test_pack.name.replace("_", "-")

    pos_rows = [r for r in test_rows if r.label == 1]
    neg_rows = [r for r in test_rows if r.label == 0]
    frame_metadata = _load_frame_metadata(roots, {r.store_id for r in test_rows}, {r.frame_id for r in test_rows})

    for size in sizes:
        for seed in seeds:
            target_neg = size // 2
            target_pos = size - target_neg

            neg_selection = _select_negative_rows(
                neg_rows,
                frame_metadata,
                seed,
                target_neg,
            )
            neg_selected = neg_selection.rows
            actual_neg = len(neg_selected)
            shortfall = max(0, target_neg - actual_neg)

            requested_pos = size - actual_neg
            pos_take = min(requested_pos, len(pos_rows))
            pos_selected = pick_k(pos_rows, seed, pos_take, key=lambda r: r.frame_id)
            actual_pos = len(pos_selected)

            if actual_neg < target_neg:
                logger.warning(
                    "Requested %d negatives but only %d available (seed=%s, size=%s)",
                    target_neg,
                    actual_neg,
                    seed,
                    size,
                )
            if actual_pos + actual_neg < size:
                logger.warning(
                    "Requested train size %d but only %d frames available for seed=%s",
                    size,
                    actual_pos + actual_neg,
                    seed,
                )

            if size >= 200:
                val_fraction = VAL_FRACTION_RULE["size_gte_200"]
            elif size >= 100:
                val_fraction = VAL_FRACTION_RULE["size_100_to_199"]
            else:
                val_fraction = VAL_FRACTION_RULE["size_lt_100"]

            def _build_units(rows: List[ManifestRow], label: int) -> list[dict[str, object]]:
                units: dict[str, dict[str, object]] = {}
                if label == 1:
                    for row in rows:
                        units[row.frame_id] = {
                            "key": row.frame_id,
                            "rows": [row],
                            "tie": row.frame_id,
                            "sequence_id": None,
                        }
                    return list(units.values())

                for row in rows:
                    info = frame_metadata.get(row.frame_id, {})
                    seq_id = info.get("sequence_id")
                    seq_key = seq_id or row.frame_id
                    unit = units.setdefault(
                        seq_key,
                        {
                            "key": seq_key,
                            "rows": [],
                            "tie": row.frame_id,
                            "sequence_id": seq_id if seq_id else None,
                        },
                    )
                    rows_bucket = cast(List[ManifestRow], unit.setdefault("rows", []))
                    rows_bucket.append(row)
                    unit["rows"] = rows_bucket
                    tie_val = cast(str, unit.setdefault("tie", row.frame_id))
                    unit["tie"] = min(tie_val, row.frame_id)
                return list(units.values())

            def _split_units_by_frames(
                units: list[dict[str, object]],
                target_frames: int,
            ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
                if not units:
                    return [], []

                ordered = sorted(
                    units,
                    key=lambda u: (
                        hmac_rank(seed, str(u["key"])),
                        str(u["tie"]),
                    ),
                )

                val_units: list[dict[str, object]] = []
                train_units: list[dict[str, object]] = []
                frames_in_val = 0

                for unit in ordered:
                    rows_u = cast(List[ManifestRow], unit["rows"])
                    if frames_in_val < target_frames:
                        val_units.append(unit)
                        frames_in_val += len(rows_u)
                    else:
                        train_units.append(unit)

                if not train_units and val_units:
                    fallback_unit = val_units.pop()
                    train_units.append(fallback_unit)

                return train_units, val_units

            pos_units = _build_units(pos_selected, 1)
            neg_units = _build_units(neg_selected, 0)

            val_pos_frames_target = int(val_fraction * len(pos_selected))
            val_neg_frames_target = int(val_fraction * len(neg_selected))

            train_pos_units, val_pos_units = _split_units_by_frames(pos_units, val_pos_frames_target)
            train_neg_units, val_neg_units = _split_units_by_frames(neg_units, val_neg_frames_target)

            train_rows = [
                replace(row, split="train")
                for unit in train_pos_units + train_neg_units
                for row in cast(List[ManifestRow], unit["rows"])
            ]
            val_rows = [
                replace(row, split="val")
                for unit in val_pos_units + val_neg_units
                for row in cast(List[ManifestRow], unit["rows"])
            ]

            selected_ids = {r.frame_id for r in train_rows + val_rows}

            selected_sequence_ids: set[str] = {
                str(unit["sequence_id"])
                for unit in train_neg_units + val_neg_units
                if unit.get("sequence_id")
            }

            excluded_sequences_in_test = sorted(selected_sequence_ids)

            sequence_frame_ids: set[str] = set()
            if selected_sequence_ids:
                for row in test_rows:
                    info = frame_metadata.get(row.frame_id, {})
                    seq_id = info.get("sequence_id")
                    if seq_id and str(seq_id) in selected_sequence_ids:
                        sequence_frame_ids.add(row.frame_id)

            test_rows_out: List[ManifestRow] = []
            for row in test_rows:
                if row.frame_id in selected_ids:
                    continue
                if row.label == 0 and row.frame_id in sequence_frame_ids:
                    continue
                test_rows_out.append(replace(row, split="test"))

            def _count_split(rows: list[ManifestRow]) -> dict[str, int]:
                total = len(rows)
                pos_ct = sum(r.label == 1 for r in rows)
                neg_ct = total - pos_ct

                neg_frame = 0
                neg_seq_frames = 0
                neg_seq_sequences: set[str] = set()

                for row in rows:
                    if row.label != 0:
                        continue
                    info = frame_metadata.get(row.frame_id, {})
                    seq_raw = info.get("sequence_id")
                    seq_id = str(seq_raw) if seq_raw else None

                    if seq_id:
                        neg_seq_frames += 1
                        neg_seq_sequences.add(seq_id)
                    else:
                        neg_frame += 1

                return {
                    "total": total,
                    "pos": pos_ct,
                    "neg": neg_ct,
                    "neg_frame": neg_frame,
                    "neg_seq_frames": neg_seq_frames,
                    "neg_seq_sequences": len(neg_seq_sequences),
                }

            def _negative_source(rows: list[ManifestRow]) -> dict[str, int]:
                base = 0
                seq = 0
                for row in rows:
                    if row.label != 0:
                        continue
                    info = frame_metadata.get(row.frame_id, {})
                    if info.get("sequence_id"):
                        seq += 1
                    else:
                        base += 1
                return {"base": base, "sequence": seq}

            train_counts = _count_split(train_rows)
            val_counts = _count_split(val_rows)
            test_counts = _count_split(test_rows_out)

            neg_source_train = _negative_source(train_rows)
            neg_source_val = _negative_source(val_rows)
            neg_source_test = _negative_source(test_rows_out)

            total_selected = len(train_rows) + len(val_rows)
            pos_val = sum(r.label == 1 for r in val_rows)
            neg_val = sum(r.label == 0 for r in val_rows)
            pos_total = sum(r.label == 1 for r in train_rows + val_rows)
            neg_total = sum(r.label == 0 for r in train_rows + val_rows)

            val_fraction_realized = {
                "overall": (len(val_rows) / total_selected) if total_selected else 0.0,
                "pos": (pos_val / pos_total) if pos_total else 0.0,
                "neg": (neg_val / neg_total) if neg_total else 0.0,
            }

            pack_dir = out_dir / f"polypgen_fewshot_s{size}_seed{seed}"

            meta = {
                "name": f"polypgen_fewshot_s{size}_seed{seed}",
                "version": "1.0.0",
                "policy": {
                    "unit_of_split": "frame",
                    "fewshot_budget_S": size,
                    "parent_pack": test_pack.name,
                    "target_train_size": size,
                    "target_class_balance": {"pos": target_pos, "neg": target_neg},
                    "class_ratio_per_split": {
                        "train": {"pos": train_counts["pos"], "neg": train_counts["neg"]},
                        "val": {"pos": val_counts["pos"], "neg": val_counts["neg"]},
                        "test": {"pos": test_counts["pos"], "neg": test_counts["neg"]},
                    },
                    "realized_train_size": len(train_rows),
                    "realized_train_counts": {
                        "pos": train_counts["pos"],
                        "neg": train_counts["neg"],
                    },
                    "balance_relaxed": actual_neg < target_neg,
                    "val_fraction_target": val_fraction,
                    "val_fraction_realized": val_fraction_realized,
                    "val_fraction_rule": dict(VAL_FRACTION_RULE),
                    "neg_sampling": {
                        "strategy": "pool_first_sequence_cap_single_pass",
                        "pool_first": True,
                        "max_frames_per_sequence": MAX_NEG_FRAMES_PER_SEQUENCE,
                        "sequence_level_disjointness": True,
                        "phase_counts": {
                            "base_pool": neg_selection.base_selected_count,
                            "sequence": actual_neg - neg_selection.base_selected_count,
                        },
                        "sequences_in_train": len({
                            unit.get("sequence_id")
                            for unit in train_neg_units
                            if unit.get("sequence_id")
                        }),
                    },
                    "negatives_source": {
                        "train": neg_source_train,
                        "val": neg_source_val,
                        "test": neg_source_test,
                    },
                    "max_frames_per_sequence_train": MAX_NEG_FRAMES_PER_SEQUENCE,
                    "sequence_disjointness_across": ["train", "val", "test"],
                    "sampling": (
                        "pos:frame_id; "
                        f"neg:frame_pool_then_sequence(cap={MAX_NEG_FRAMES_PER_SEQUENCE}); "
                        f"seed={seed}"
                    ),
                },
                "row_schema": {
                    "fields": [
                        {"name": "dataset", "type": "string", "desc": "constant 'polypgen_clean'"},
                        {"name": "split", "type": "string", "desc": "train|val|test (frame-level)"},
                        {"name": "case_id", "type": "string"},
                        {"name": "frame_id", "type": "string"},
                        {"name": "frame_path", "type": "path"},
                        {"name": "label", "type": "int"},
                    ]
                },
                "selection_keys": {
                    "train": f"hmac_sha256(frame_id,{seed})",
                    "val": f"hmac_sha256(unit_key=frame_id|sequence_id_or_frame_id,{seed})",
                    "test": (
                        f"complement of train|val with neg-sequence exclusion within {parent_pack_label}"
                    ),
                },
                "counts": {
                    "train": train_counts,
                    "val": val_counts,
                    "test": test_counts,
                },
                "notes": {
                    "excluded_sequences_in_test": excluded_sequences_in_test,
                },
                "roots": roots,
            }

            if shortfall > 0:
                neg_sampling_policy = cast(dict[str, object], meta["policy"]["neg_sampling"])  # type: ignore[index]
                neg_sampling_policy["shortfall"] = shortfall

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

            train_ids = {r.frame_id for r in train_rows}
            val_ids = {r.frame_id for r in val_rows}
            test_ids = {r.frame_id for r in test_rows_out}

            assert train_ids.isdisjoint(val_ids), "train/val frame overlap"
            assert train_ids.isdisjoint(test_ids), "train/test frame overlap"
            assert val_ids.isdisjoint(test_ids), "val/test frame overlap"

            assert not any(
                frame_metadata.get(r.frame_id, {}).get("sequence_id") in selected_sequence_ids
                for r in test_rows_out
                if r.label == 0
            ), "test split includes selected negative sequence"

            write_pack(
                pack_dir,
                {"train": train_rows, "val": val_rows, "test": test_rows_out},
                meta,
            )


__all__ = ["build_polypgen_fewshot"]
