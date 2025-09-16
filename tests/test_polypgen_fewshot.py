from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import polypdb.fewshot.polypgen_fewshot as pg_fewshot
from polypdb.fewshot import build_polypgen_fewshot
from polypdb.manifests.core import ManifestRow, hmac_rank, pick_k
from polypdb.manifests.io import read_csv, write_pack


def _make_test_pack(tmp_path: Path) -> tuple[Path, List[Dict[str, object]]]:
    root = tmp_path / "polypgen_root"
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    frame_records: list[dict[str, object]] = []

    # Positive frames with known centres
    for i in range(6):
        frame_id = f"pos{i}"
        frame_path = f"images/{frame_id}.png"
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="test",
                case_id=f"C{i%2}",
                frame_id=frame_id,
                frame_path=frame_path,
                label=1,
                store_id="polypgen_clean",
            )
        )
        frame_records.append(
            {
                "global_frame_id": frame_id,
                "img_path": frame_path,
                "has_polyp": True,
                "center_id": f"C{i%2}",
                "origin": "frame",
                "sequence_id": None,
            }
        )

    # Base negatives with known centres
    base_neg_ids = ["neg_base0", "neg_base1"]
    for idx, frame_id in enumerate(base_neg_ids):
        frame_path = f"images/{frame_id}.png"
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="test",
                case_id=f"NEG_CASE_{idx}",
                frame_id=frame_id,
                frame_path=frame_path,
                label=0,
                store_id="polypgen_clean",
            )
        )
        frame_records.append(
            {
                "global_frame_id": frame_id,
                "img_path": frame_path,
                "has_polyp": False,
                "center_id": f"NEG{idx}",
                "origin": "frame",
                "sequence_id": None,
            }
        )

    # Sequence negatives without centre information
    seq_layout = {"seqA_neg": 4, "seqB_neg": 3}
    for seq_id, count in seq_layout.items():
        for idx in range(count):
            frame_id = f"{seq_id}_f{idx}"
            frame_path = f"images/{frame_id}.png"
            rows.append(
                ManifestRow(
                    dataset="PolypGen",
                    split="test",
                    case_id=f"NEGSEQ_{seq_id}",
                    frame_id=frame_id,
                    frame_path=frame_path,
                    label=0,
                    store_id="polypgen_clean",
                )
            )
            frame_records.append(
                {
                    "global_frame_id": frame_id,
                    "img_path": frame_path,
                    "has_polyp": False,
                    "center_id": None,
                    "origin": "neg_seq",
                    "sequence_id": seq_id,
                }
            )

    pack_dir = tmp_path / "test_pack"
    meta = {
        "name": "polypgen-clean-test",
        "version": "0.0",
        "policy": {"unit_of_split": "frame"},
        "roots": {"polypgen_clean": str(root)},
    }
    write_pack(pack_dir, {"test": rows}, meta)
    return pack_dir, frame_records


def test_polypgen_fewshot_disjoint_and_balanced(tmp_path: Path, monkeypatch) -> None:
    test_pack, frame_records = _make_test_pack(tmp_path)
    metadata = {
        rec["global_frame_id"]: {
            "origin": rec["origin"],
            "sequence_id": rec["sequence_id"],
        }
        for rec in frame_records
    }
    monkeypatch.setattr(
        pg_fewshot,
        "_load_frame_metadata",
        lambda roots, store_ids, frame_ids: {
            fid: metadata[fid]
            for fid in frame_ids
            if fid in metadata
        },
    )

    out_dir = tmp_path / "out"
    build_polypgen_fewshot(test_pack, [6], [123], out_dir)

    pack_dir = out_dir / "polypgen_fewshot_s6_seed123"
    train_rows = read_csv(pack_dir / "train.csv")
    test_rows = read_csv(pack_dir / "test.csv")

    # Disjointness at frame level
    train_ids = {r.frame_id for r in train_rows}
    test_ids = {r.frame_id for r in test_rows}
    assert train_ids.isdisjoint(test_ids)

    # Ensure base negatives are prioritised
    neg_rows_all = [r for r in read_csv(test_pack / "test.csv") if r.label == 0]
    base_negatives = [r for r in neg_rows_all if metadata.get(r.frame_id, {}).get("origin") != "neg_seq"]
    expected_base = pick_k(base_negatives, 123, min(3, len(base_negatives)), key=lambda r: r.frame_id)
    train_neg_ids = {r.frame_id for r in train_rows if r.label == 0}
    assert {r.frame_id for r in expected_base}.issubset(train_neg_ids)

    # Determine expected sequence contributions
    remaining = 3 - len(expected_base)
    sequences: dict[str, list[ManifestRow]] = {}
    for row in neg_rows_all:
        seq_id = metadata.get(row.frame_id, {}).get("sequence_id")
        if seq_id:
            sequences.setdefault(seq_id, []).append(row)

    expected_sequence_ids: list[str] = []
    if remaining > 0:
        ordered_seq_ids = sorted(sequences, key=lambda sid: (hmac_rank(123, sid), sid))
        for seq_id in ordered_seq_ids:
            seq_rows = sorted(sequences[seq_id], key=lambda r: (hmac_rank(123, r.frame_id), r.frame_id))
            take = min(pg_fewshot.MAX_NEG_FRAMES_PER_SEQUENCE, remaining, len(seq_rows))
            if take <= 0:
                continue
            expected_sequence_ids.append(seq_id)
            remaining -= take
            if remaining <= 0:
                break

    selected_sequence_ids = {
        metadata.get(r.frame_id, {}).get("sequence_id")
        for r in train_rows
        if r.label == 0 and metadata.get(r.frame_id, {}).get("sequence_id")
    }
    assert set(expected_sequence_ids) == selected_sequence_ids

    test_sequence_ids = {
        metadata.get(r.frame_id, {}).get("sequence_id")
        for r in test_rows
        if r.label == 0 and metadata.get(r.frame_id, {}).get("sequence_id")
    }
    assert not selected_sequence_ids & test_sequence_ids

    # Approximate balance
    pos = sum(r.label == 1 for r in train_rows)
    neg = sum(r.label == 0 for r in train_rows)
    assert pos + neg == 6
    assert abs(pos - neg) <= 1

    manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text())
    assert manifest["policy"]["selection"] == "hmac_sha256(frame_id,123)"
    assert manifest["counts"]["train"]["frames"] == 6
    assert manifest["counts"]["test"]["frames"] == len(test_rows)
