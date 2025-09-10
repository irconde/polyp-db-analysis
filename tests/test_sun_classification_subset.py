from pathlib import Path

import pandas as pd
from PIL import Image

from polypdb.sun_classification import (
    prepare_sun_classification,
    create_sun_classification_subset,
)


def _create_canonical_dataset(
    root: Path,
    n_pos: int = 10,
    n_neg_cases: int = 10,
    frames_per_case: int = 33,
) -> Path:
    images_dir = root / "images"
    manifests_dir = root / "manifests"
    images_dir.mkdir(parents=True)
    manifests_dir.mkdir()

    records = []
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
    neg_case_id = "N0"
    for f in range(n_neg_cases * frames_per_case):
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
    case_records = {"global_case_id": [f"P{i}" for i in range(n_pos)] + [neg_case_id]}
    pd.DataFrame(case_records).to_parquet(manifests_dir / "cases.parquet", index=False)
    return root


def _case_ids(root: Path, split: str, cls: str) -> set[str]:
    return {p.stem.split("_")[0] for p in (root / split / cls).glob("*.jpg")}


def test_create_subset(tmp_path: Path) -> None:
    canonical = _create_canonical_dataset(tmp_path / "canonical")
    master = tmp_path / "master"
    prepare_sun_classification(canonical, master, seed=0)

    subset1 = tmp_path / "subset1"
    create_sun_classification_subset(master, subset1, fraction=0.5, seed=0)
    subset2 = tmp_path / "subset2"
    create_sun_classification_subset(master, subset2, fraction=0.5, seed=0)

    assert _case_ids(subset1, "train", "polyp") == _case_ids(
        subset2, "train", "polyp"
    )
    assert _case_ids(subset1, "train", "non-polyp") == _case_ids(
        subset2, "train", "non-polyp"
    )

    subset3 = tmp_path / "subset3"
    create_sun_classification_subset(master, subset3, fraction=0.5, seed=1)

    assert (
        _case_ids(subset1, "train", "polyp")
        != _case_ids(subset3, "train", "polyp")
        or _case_ids(subset1, "train", "non-polyp")
        != _case_ids(subset3, "train", "non-polyp")
    )

    for split in ["val", "test"]:
        for cls in ["polyp", "non-polyp"]:
            src_files = {p.name for p in (master / split / cls).glob("*.jpg")}
            dst_files = {p.name for p in (subset1 / split / cls).glob("*.jpg")}
            assert src_files == dst_files

    n_train_cases = len(_case_ids(master, "train", "polyp"))
    expected = max(1, round(0.5 * n_train_cases))
    assert len(_case_ids(subset1, "train", "polyp")) == expected
    assert len(_case_ids(subset1, "train", "non-polyp")) == expected
