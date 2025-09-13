from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
import yaml

from polypdb.datasets.sun_build import build_sun_morphology
from polypdb.manifests.io import read_csv


def _create_canonical_dataset(
    root: Path,
    n_pos: int = 14,
    n_neg_cases: int = 14,
    frames_per_case: int = 33,
    paris_classes: list[str] | None = None,
) -> Path:
    images_dir = root / "images"
    manifests_dir = root / "manifests"
    images_dir.mkdir(parents=True)
    manifests_dir.mkdir()

    records = []
    # positive cases
    for c in range(n_pos):
        case_id = f"P{c}"
        for f in range(frames_per_case):
            fname = f"{case_id}.{f:06d}.jpg"
            Image.new("RGB", (1, 1)).save(images_dir / fname)
            records.append(
                {
                    "global_frame_id": fname[:-4],
                    "global_case_id": case_id,
                    "has_polyp": True,
                    "img_path": f"images/{fname}",
                }
            )
    # negative cases
    for n in range(n_neg_cases):
        neg_case_id = f"N{n}"
        for f in range(frames_per_case):
            fname = f"{neg_case_id}.{f:06d}.jpg"
            Image.new("RGB", (1, 1)).save(images_dir / fname)
            records.append(
                {
                    "global_frame_id": fname[:-4],
                    "global_case_id": neg_case_id,
                    "has_polyp": False,
                    "img_path": f"images/{fname}",
                }
            )

    frames_df = pd.DataFrame(records)
    frames_df.to_parquet(manifests_dir / "frames.parquet", index=False)
    case_records = {
        "global_case_id": [f"P{i}" for i in range(n_pos)]
        + [f"N{n}" for n in range(n_neg_cases)]
    }
    if paris_classes:
        case_records["paris_class"] = paris_classes + [None] * n_neg_cases
    pd.DataFrame(case_records).to_parquet(manifests_dir / "cases.parquet", index=False)
    return root


def _case_morph_counts(rows):
    mapping = {}
    for r in rows:
        if r.label == 1:
            mapping.setdefault(r.case_id, r.morphology)
    return Counter(mapping.values())


def test_build_sun_morphology(tmp_path: Path) -> None:
    paris = ["Is"] * 7 + ["IIa"] * 7
    canonical = _create_canonical_dataset(
        tmp_path / "canonical", n_pos=14, n_neg_cases=14, paris_classes=paris
    )
    out_dir = tmp_path / "pack"
    build_sun_morphology(canonical, out_dir, seed=0, verify=False)

    splits = {s: read_csv(out_dir / f"{s}.csv") for s in ("train", "val", "test")}

    case_sets = [set(r.case_id for r in rows) for rows in splits.values()]
    for i in range(len(case_sets)):
        for j in range(i + 1, len(case_sets)):
            assert case_sets[i].isdisjoint(case_sets[j])

    for rows in splits.values():
        counts = Counter(r.case_id for r in rows)
        assert counts and all(v == 33 for v in counts.values())

    def count_cases(rows, label):
        return len({r.case_id for r in rows if r.label == label})

    assert count_cases(splits["train"], 1) == count_cases(splits["train"], 0) == 10
    assert count_cases(splits["val"], 1) == count_cases(splits["val"], 0) == 2
    assert count_cases(splits["test"], 1) == count_cases(splits["test"], 0) == 2

    assert _case_morph_counts(splits["train"]) == {"polypoid": 5, "flat": 5}
    assert _case_morph_counts(splits["val"]) == {"polypoid": 1, "flat": 1}
    assert _case_morph_counts(splits["test"]) == {"polypoid": 1, "flat": 1}

    for rows in splits.values():
        for r in rows:
            if r.label == 0:
                assert r.split_unit == "source_case"
                assert r.source_case_id.startswith("N")
                assert r.synthetic_case_id

    manifest = yaml.safe_load((out_dir / "manifest.yaml").read_text())
    mmap = manifest["morphology_map"]
    assert Counter(mmap.values()) == {"polypoid": 7, "flat": 7}
    counts = manifest["morphology_counts"]
    assert counts["overall"] == {
        "flat_cases": 7,
        "polypoid_cases": 7,
        "flat_frames": 7 * 33,
        "polypoid_frames": 7 * 33,
    }
    assert counts["train"] == {
        "flat_cases": 5,
        "polypoid_cases": 5,
        "flat_frames": 5 * 33,
        "polypoid_frames": 5 * 33,
    }
    assert counts["val"] == {
        "flat_cases": 1,
        "polypoid_cases": 1,
        "flat_frames": 33,
        "polypoid_frames": 33,
    }
    assert counts["test"] == {
        "flat_cases": 1,
        "polypoid_cases": 1,
        "flat_frames": 33,
        "polypoid_frames": 33,
    }
    assert manifest["assertions"]["morphology_targets_met"] == {
        "train": {"flat": 5, "polypoid": 5},
        "val": {"flat": 1, "polypoid": 1},
        "test": {"flat": 1, "polypoid": 1},
    }

