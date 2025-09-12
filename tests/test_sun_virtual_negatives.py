"""Tests for virtual negative case generation in :func:`build_sun_full`."""

from collections import Counter
from pathlib import Path

from polypdb.datasets.sun_build import build_sun_full
from polypdb.manifests.io import read_csv


def _make_index():
    index = {}
    # 100 positive cases each with exactly 33 frames
    for i in range(100):
        cid = f"p{i:03d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(33)
        ]
        index[cid] = {"label": 1, "frames": frames}

    # 13 long negative cases each with 330 frames (10 chunks of 33)
    for i in range(13):
        cid = f"n{i:02d}"
        frames = [
            {"frame_path": f"{cid}/f{j:04d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(330)
        ]
        index[cid] = {"label": 0, "frames": frames}

    return index


def test_virtual_negative_cases(tmp_path):
    index = _make_index()

    def indexer(_root: Path):
        return index

    build_sun_full(Path("/data"), tmp_path, seed=0, indexer=indexer)

    splits = {split: read_csv(tmp_path / f"{split}.csv") for split in ["train", "val", "test"]}

    for split, expected in [("train", 70), ("val", 15), ("test", 15)]:
        rows = splits[split]
        pos_cases = {r.case_id for r in rows if r.label == 1}
        neg_cases = {r.case_id for r in rows if r.label == 0}
        assert len(pos_cases) == expected
        assert len(neg_cases) == expected
        counts = Counter(r.case_id for r in rows)
        assert all(v == 33 for v in counts.values())

    all_neg_cases = {
        r.case_id for rows in splits.values() for r in rows if r.label == 0
    }
    assert len(all_neg_cases) == 100

