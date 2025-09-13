from pathlib import Path
from collections import Counter

from polypdb.datasets.sun_build import build_sun_full
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
