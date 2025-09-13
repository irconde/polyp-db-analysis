from pathlib import Path
from collections import Counter

import pytest

from polypdb.datasets.sun_build import build_sun_full, _compute_assertions, _split_targets
from polypdb.manifests.core import ManifestRow
from polypdb.manifests.io import read_csv


def _make_index():
    index = {}
    # three positive cases
    for i in range(3):
        cid = f"p{i:02d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(40)
        ]
        index[cid] = {"label": 1, "frames": frames}
    # two negative cases
    for i in range(2):
        cid = f"n{i:02d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(40)
        ]
        index[cid] = {"label": 0, "frames": frames}
    return index


def test_case_disjoint_and_frame_count(tmp_path):
    index = _make_index()

    def indexer(_root: Path):
        return index

    build_sun_full(Path("/data"), tmp_path, seed=42, indexer=indexer, verify=False)

    splits = {}
    for split in ["train", "val", "test"]:
        csv = tmp_path / f"{split}.csv"
        if csv.exists():
            splits[split] = read_csv(csv)
        else:
            splits[split] = []

    # ensure each case appears only in one split
    case_sets = [set(r.case_id for r in rows) for rows in splits.values()]
    for i in range(len(case_sets)):
        for j in range(i + 1, len(case_sets)):
            assert case_sets[i].isdisjoint(case_sets[j])

    # each case has exactly 33 frames selected
    counts = Counter()
    for rows in splits.values():
        counts.update(r.case_id for r in rows)
    assert counts and all(v == 33 for v in counts.values())


def _row(case_id: str, label: int, split: str) -> ManifestRow:
    return ManifestRow(
        dataset="sun",
        split=split,
        case_id=case_id,
        frame_id=f"{case_id}_0",
        frame_path=f"{case_id}/0.jpg",
        label=label,
        store_id="",
        source_case_id=case_id,
    )


def test_dynamic_split_targets_accept_non_100_cases():
    frames_per_case = 1
    splits = {"train": [], "val": [], "test": []}

    # 10 positive and 10 negative cases
    for i in range(10):
        cid = f"p{i:02d}"
        split = "train" if i < 7 else "val" if i < 9 else "test"
        splits[split].append(_row(cid, 1, split))
    for i in range(10):
        cid = f"n{i:02d}"
        split = "train" if i < 7 else "val" if i < 9 else "test"
        splits[split].append(_row(cid, 0, split))

    assertions = _compute_assertions(splits, frames_per_case, verify=True)

    pos_targets = _split_targets(10)
    neg_targets = _split_targets(10)
    expected = {
        split: {
            "pos_cases": pos_targets[split],
            "neg_cases": neg_targets[split],
            "frames": frames_per_case * (pos_targets[split] + neg_targets[split]),
        }
        for split in ["train", "val", "test"]
    }

    assert assertions["split_targets_met"] == expected
    assert "split_target_mismatches" not in assertions


def test_dynamic_split_targets_mismatch_raises():
    frames_per_case = 1
    splits = {"train": [], "val": [], "test": []}

    # 10 pos/neg cases but skewed split: train has 8 each, val/test have 1 each
    for i in range(10):
        cid = f"p{i:02d}"
        split = "train" if i < 8 else "val" if i == 8 else "test"
        splits[split].append(_row(cid, 1, split))
    for i in range(10):
        cid = f"n{i:02d}"
        split = "train" if i < 8 else "val" if i == 8 else "test"
        splits[split].append(_row(cid, 0, split))

    with pytest.raises(AssertionError):
        _compute_assertions(splits, frames_per_case, verify=True)
