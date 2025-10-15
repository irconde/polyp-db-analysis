from __future__ import annotations

from pathlib import Path
import hashlib
import sys
from typing import Dict, List

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import polypdb.fewshot.polypgen_fewshot as pg_fewshot
from polypdb.fewshot import build_polypgen_fewshot, build_polypgen_fewshot_canonical
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


def _expected_alpha(size: int) -> float:
    if size >= 200:
        return pg_fewshot.VAL_FRACTION_RULE["size_gte_200"]
    if size >= 100:
        return pg_fewshot.VAL_FRACTION_RULE["size_100_to_199"]
    return pg_fewshot.VAL_FRACTION_RULE["size_lt_100"]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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

    pack_dir = out_dir / "polypgen_fewshot_s6"
    train_rows = read_csv(pack_dir / "train.csv")
    val_rows = read_csv(pack_dir / "val.csv")
    test_rows = read_csv(pack_dir / "test.csv")

    # Disjointness at frame level across splits
    train_ids = {r.frame_id for r in train_rows}
    val_ids = {r.frame_id for r in val_rows}
    test_ids = {r.frame_id for r in test_rows}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Ensure base negatives are prioritised
    neg_rows_all = [r for r in read_csv(test_pack / "test.csv") if r.label == 0]
    base_negatives = [r for r in neg_rows_all if metadata.get(r.frame_id, {}).get("origin") != "neg_seq"]
    expected_base = pick_k(base_negatives, 123, min(3, len(base_negatives)), key=lambda r: r.frame_id)
    train_neg_ids = {r.frame_id for r in train_rows if r.label == 0}
    train_val_neg_ids = train_neg_ids | {r.frame_id for r in val_rows if r.label == 0}
    assert {r.frame_id for r in expected_base}.issubset(train_val_neg_ids)

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
        for r in train_rows + val_rows
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
    pos_train = sum(r.label == 1 for r in train_rows)
    neg_train = sum(r.label == 0 for r in train_rows)
    pos_val = sum(r.label == 1 for r in val_rows)
    neg_val = sum(r.label == 0 for r in val_rows)
    assert pos_train + neg_train == len(train_rows)
    assert pos_val + neg_val == len(val_rows)

    alpha = _expected_alpha(6)
    for label, total, val_count in (
        (1, pos_train + pos_val, pos_val),
        (0, neg_train + neg_val, neg_val),
    ):
        if total == 0:
            continue
        realized = val_count / total
        tolerance = max(0.15, 1 / total)
        assert realized <= alpha + tolerance + 1e-6
        assert realized + tolerance + 1e-6 >= max(0, alpha - tolerance)

    assert abs((pos_train + pos_val) - (neg_train + neg_val)) <= 1
    assert abs(pos_train - neg_train) <= 1
    if val_rows:
        assert abs(pos_val - neg_val) <= 1

    manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text())
    policy = manifest["policy"]
    assert manifest["name"] == "polypgen_fewshot_s6"
    assert policy["parent_pack"] == test_pack.name
    assert policy["target_train_size"] == 6
    assert policy["fewshot_budget_S"] == 6
    assert policy["target_class_balance"] == {"pos": 3, "neg": 3}
    assert policy["realized_train_size"] == len(train_rows)
    assert policy["realized_train_counts"] == {"pos": pos_train, "neg": neg_train}
    assert policy["balance_relaxed"] is False
    assert policy["val_fraction_target"] == alpha
    assert policy["val_fraction_realized"]["overall"] == pytest.approx(
        len(val_rows) / (len(train_rows) + len(val_rows)) if (len(train_rows) + len(val_rows)) else 0.0
    )
    assert policy["val_fraction_rule"] == pg_fewshot.VAL_FRACTION_RULE
    assert policy["class_ratio_per_split"]["train"] == {"pos": pos_train, "neg": neg_train}
    assert policy["class_ratio_per_split"]["val"] == {"pos": pos_val, "neg": neg_val}
    assert policy["class_ratio_per_split"]["test"] == {
        "pos": sum(r.label == 1 for r in test_rows),
        "neg": sum(r.label == 0 for r in test_rows),
    }
    assert policy["negatives_source"]["train"] == {
        "base": sum(
            1 for r in train_rows if r.label == 0 and not metadata.get(r.frame_id, {}).get("sequence_id")
        ),
        "sequence": sum(
            1 for r in train_rows if r.label == 0 and metadata.get(r.frame_id, {}).get("sequence_id")
        ),
    }
    assert policy["negatives_source"]["val"] == {
        "base": sum(
            1 for r in val_rows if r.label == 0 and not metadata.get(r.frame_id, {}).get("sequence_id")
        ),
        "sequence": sum(
            1 for r in val_rows if r.label == 0 and metadata.get(r.frame_id, {}).get("sequence_id")
        ),
    }
    assert policy["negatives_source"]["test"] == {
        "base": sum(
            1 for r in test_rows if r.label == 0 and not metadata.get(r.frame_id, {}).get("sequence_id")
        ),
        "sequence": sum(
            1 for r in test_rows if r.label == 0 and metadata.get(r.frame_id, {}).get("sequence_id")
        ),
    }
    assert policy["max_frames_per_sequence_train"] == pg_fewshot.MAX_NEG_FRAMES_PER_SEQUENCE
    assert policy["sequence_disjointness_across"] == ["train", "val", "test"]
    assert str(seed := manifest["generator"]["seed"]) == "123"
    expected_sampling = (
        "pos:frame_id; "
        f"neg:frame_pool_then_sequence(cap={pg_fewshot.MAX_NEG_FRAMES_PER_SEQUENCE}); "
        "seed=123"
    )
    assert policy["sampling"] == expected_sampling
    assert "selection" not in policy

    neg_policy = policy["neg_sampling"]
    assert neg_policy["strategy"] == "pool_first_sequence_cap_single_pass"
    assert neg_policy["pool_first"] is True
    assert neg_policy["max_frames_per_sequence"] == pg_fewshot.MAX_NEG_FRAMES_PER_SEQUENCE
    assert neg_policy["sequence_level_disjointness"] is True
    total_neg_selected = neg_train + neg_val
    assert neg_policy["phase_counts"]["base_pool"] == len(expected_base)
    assert neg_policy["phase_counts"]["sequence"] == total_neg_selected - len(expected_base)
    train_sequence_ids = {
        metadata.get(r.frame_id, {}).get("sequence_id")
        for r in train_rows
        if r.label == 0 and metadata.get(r.frame_id, {}).get("sequence_id")
    }
    assert neg_policy["sequences_in_train"] == len(train_sequence_ids)
    assert neg_policy.get("shortfall", 0) == 0

    parent_manifest = yaml.safe_load((test_pack / "manifest.yaml").read_text())
    parent_name = parent_manifest["name"]
    assert manifest["selection_keys"]["train"] == "hmac_sha256(frame_id,123)"
    assert "unit_key=frame_id|sequence_id_or_frame_id" in manifest["selection_keys"]["val"]
    assert manifest["selection_keys"]["test"] == (
        f"complement of train|val with neg-sequence exclusion within {parent_name}"
    )

    assert manifest["row_schema"]["fields"][1]["desc"] == "train|val|test (frame-level)"

    counts = manifest["counts"]

    def _expected_neg_breakdown(rows: list[ManifestRow]) -> tuple[int, int, int]:
        neg_frame = 0
        neg_seq_frames = 0
        neg_seq_sequences: set[str] = set()
        for row in rows:
            if row.label != 0:
                continue
            info = metadata.get(row.frame_id, {})
            seq_raw = info.get("sequence_id")
            seq_id = str(seq_raw) if seq_raw else None
            if seq_id:
                neg_seq_frames += 1
                neg_seq_sequences.add(seq_id)
            else:
                neg_frame += 1
        return neg_frame, neg_seq_frames, len(neg_seq_sequences)

    for split_name, split_rows in (
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ):
        neg_frame_expected, neg_seq_frames_expected, neg_seq_sequences_expected = _expected_neg_breakdown(split_rows)
        assert counts[split_name]["frames"] == len(split_rows)
        assert counts[split_name]["pos"] == sum(r.label == 1 for r in split_rows)
        assert counts[split_name]["neg"] == sum(r.label == 0 for r in split_rows)
        assert counts[split_name]["neg_frame"] == neg_frame_expected
        assert counts[split_name]["neg_seq_frames"] == neg_seq_frames_expected
        assert counts[split_name]["neg_seq_sequences"] == neg_seq_sequences_expected

    notes = manifest["notes"]
    assert notes["excluded_sequences_in_test"] == sorted(expected_sequence_ids)

    # Determinism check via hashes
    hashes_first = {
        split: _hash_file(pack_dir / f"{split}.csv") for split in ("train", "val", "test")
    }
    build_polypgen_fewshot(test_pack, [6], [123], out_dir)
    hashes_second = {
        split: _hash_file(pack_dir / f"{split}.csv") for split in ("train", "val", "test")
    }
    assert hashes_first == hashes_second


def test_polypgen_fewshot_relaxes_balance_when_negatives_insufficient(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "polypgen_root_small"
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    metadata: dict[str, dict[str, object | None]] = {}

    for idx in range(4):
        frame_id = f"pos{idx}"
        frame_path = f"images/{frame_id}.png"
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="test",
                case_id=f"P{idx}",
                frame_id=frame_id,
                frame_path=frame_path,
                label=1,
                store_id="polypgen_clean",
            )
        )
        metadata[frame_id] = {"origin": "frame", "sequence_id": None}

    neg_frame = "neg_only"
    rows.append(
        ManifestRow(
            dataset="PolypGen",
            split="test",
            case_id="N0",
            frame_id=neg_frame,
            frame_path=f"images/{neg_frame}.png",
            label=0,
            store_id="polypgen_clean",
        )
    )
    metadata[neg_frame] = {"origin": "frame", "sequence_id": None}

    pack_dir = tmp_path / "small_pack"
    meta = {
        "name": "polypgen-clean-test",
        "version": "0.0",
        "policy": {"unit_of_split": "frame"},
        "roots": {"polypgen_clean": str(root)},
    }
    write_pack(pack_dir, {"test": rows}, meta)

    monkeypatch.setattr(
        pg_fewshot,
        "_load_frame_metadata",
        lambda roots, store_ids, frame_ids: {
            fid: metadata[fid]
            for fid in frame_ids
            if fid in metadata
        },
    )

    out_dir = tmp_path / "out_small"
    build_polypgen_fewshot(pack_dir, [4], [7], out_dir)

    pack_dir_out = out_dir / "polypgen_fewshot_s4"
    train_rows = read_csv(pack_dir_out / "train.csv")
    val_rows = read_csv(pack_dir_out / "val.csv")

    pos_count = sum(r.label == 1 for r in train_rows)
    neg_count = sum(r.label == 0 for r in train_rows)
    assert len(train_rows) == 4
    assert pos_count == 3
    assert neg_count == 1
    assert not val_rows  # Validation should be empty due to minimum-train guard

    manifest = yaml.safe_load((pack_dir_out / "manifest.yaml").read_text())
    policy = manifest["policy"]
    assert manifest["name"] == "polypgen_fewshot_s4"
    assert policy["target_class_balance"] == {"pos": 2, "neg": 2}
    assert policy["balance_relaxed"] is True
    assert policy["realized_train_counts"] == {"pos": pos_count, "neg": neg_count}
    assert policy["realized_train_size"] == len(train_rows)
    assert policy["val_fraction_target"] == _expected_alpha(4)
    assert policy["class_ratio_per_split"]["val"] == {"pos": 0, "neg": 0}
    assert manifest["counts"]["val"]["frames"] == 0
    assert policy["val_fraction_rule"] == pg_fewshot.VAL_FRACTION_RULE

    neg_policy = policy["neg_sampling"]
    assert neg_policy["phase_counts"]["base_pool"] == 1
    assert neg_policy["phase_counts"]["sequence"] == 0
    assert neg_policy["sequences_in_train"] == 0
    assert neg_policy["shortfall"] == 1

    assert manifest["counts"]["train"]["frames"] == len(train_rows)
    assert manifest["notes"]["excluded_sequences_in_test"] == []


def test_polypgen_fewshot_multi_seed_keeps_suffix(tmp_path: Path, monkeypatch) -> None:
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

    out_dir = tmp_path / "multi"
    seeds = [5, 9]
    build_polypgen_fewshot(test_pack, [6], seeds, out_dir)

    for seed in seeds:
        pack_dir = out_dir / f"polypgen_fewshot_s6_seed{seed}"
        assert pack_dir.exists()
        manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text())
        assert manifest["name"] == f"polypgen_fewshot_s6_seed{seed}"
        assert manifest["generator"]["seed"] == seed


def test_polypgen_fewshot_canonical_builds_expected_packs(
    tmp_path: Path, monkeypatch
) -> None:
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

    out_dir = tmp_path / "canonical"
    build_polypgen_fewshot_canonical(test_pack, out_dir)

    expected_map = {
        size: pg_fewshot.CANONICAL_FEWSHOT_SEED for size in pg_fewshot.CANONICAL_FEWSHOT_SIZES
    }
    for size, seed in expected_map.items():
        pack_dir = out_dir / f"polypgen_fewshot_s{size}"
        assert pack_dir.exists()
        manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text())
        generator = manifest["generator"]
        assert manifest["name"] == f"polypgen_fewshot_s{size}"
        assert generator["seed"] == seed
        assert generator.get("size_seed_map") == expected_map
        assert "fewshot-canonical" in generator["command"]
        assert (pack_dir / "train.csv").exists()
        assert (pack_dir / "val.csv").exists()
        assert (pack_dir / "test.csv").exists()
