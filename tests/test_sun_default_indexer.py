from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image

from polypdb.datasets.sun_build import build_sun_full
from polypdb.manifests.io import read_csv


def _create_canonical_dataset(root: Path) -> Path:
    images_dir = root / "images"
    manifests_dir = root / "manifests"
    images_dir.mkdir(parents=True)
    manifests_dir.mkdir()

    records = []
    case_records = []
    # three positive cases
    for i in range(3):
        cid = f"p{i:02d}"
        shape = "flat" if i == 1 else "sessile"
        case_records.append({"global_case_id": cid, "shape": shape})
        for j in range(40):
            fname = f"{cid}.{j:06d}.jpg"
            Image.new("RGB", (1, 1)).save(images_dir / fname)
            records.append(
                {
                    "global_frame_id": fname[:-4],
                    "global_case_id": cid,
                    "has_polyp": True,
                    "img_path": f"images/{fname}",
                    "WIDTH": 1,
                    "HEIGHT": 1,
                }
            )
    # two negative cases
    for i in range(2):
        cid = f"n{i:02d}"
        case_records.append({"global_case_id": cid, "shape": None})
        for j in range(40):
            fname = f"{cid}.{j:06d}.jpg"
            Image.new("RGB", (1, 1)).save(images_dir / fname)
            records.append(
                {
                    "global_frame_id": fname[:-4],
                    "global_case_id": cid,
                    "has_polyp": False,
                    "img_path": f"images/{fname}",
                    "WIDTH": 1,
                    "HEIGHT": 1,
                }
            )

    pd.DataFrame(records).to_parquet(manifests_dir / "frames.parquet", index=False)
    pd.DataFrame(case_records).to_parquet(manifests_dir / "cases.parquet", index=False)
    return root


def test_default_indexer_build(tmp_path):
    canonical = _create_canonical_dataset(tmp_path / "canonical")
    out_dir = tmp_path / "pack"
    build_sun_full(canonical, out_dir, seed=0)

    splits = {}
    for split in ["train", "val", "test"]:
        csv = out_dir / f"{split}.csv"
        splits[split] = read_csv(csv) if csv.exists() else []

    case_sets = [set(r.case_id for r in rows) for rows in splits.values()]
    for i in range(len(case_sets)):
        for j in range(i + 1, len(case_sets)):
            assert case_sets[i].isdisjoint(case_sets[j])

    counts = Counter()
    for rows in splits.values():
        counts.update(r.case_id for r in rows)
    assert counts and all(v == 33 for v in counts.values())

    expected = {"p00": "polypoid", "p01": "flat", "p02": "polypoid"}
    found: dict[str, set[str]] = {}
    for rows in splits.values():
        for r in rows:
            if r.case_id in expected:
                found.setdefault(r.case_id, set()).add(r.morphology)
    for cid, morph in expected.items():
        assert found.get(cid) == {morph}
