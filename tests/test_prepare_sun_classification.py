from pathlib import Path
import pandas as pd
from PIL import Image

from polypdb.sun_classification import prepare_sun_classification


def _create_canonical_dataset(
    root: Path,
    n_pos: int = 7,
    n_neg_cases: int = 7,
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
        paris = paris_classes[c] if paris_classes else None
        for f in range(frames_per_case):
            fname = f"{case_id}.{f:06d}.jpg"
            Image.new("RGB", (1, 1)).save(images_dir / fname)
            records.append(
                {
                    "global_frame_id": fname[:-4],
                    "global_case_id": case_id,
                    "has_polyp": True,
                    "img_path": f"images/{fname}",
                    # paris_class will be merged from cases.parquet if provided
                }
            )
    # single long negative video
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
    if paris_classes:
        case_records["paris_class"] = paris_classes + [None]
    pd.DataFrame(case_records).to_parquet(manifests_dir / "cases.parquet", index=False)
    return root


def test_prepare_sun_classification(tmp_path):
    canonical = _create_canonical_dataset(tmp_path / "canonical", n_pos=7, n_neg_cases=14)
    out_dir = tmp_path / "out"
    prepare_sun_classification(canonical, out_dir, seed=0)

    def count(split: str, cls: str) -> int:
        return len(list((out_dir / split / cls).glob("*.jpg")))

    # expected case counts after balancing positives and negatives
    n_cases = min(7, 14)
    train_cases = int(n_cases * 0.7)
    val_cases = int(n_cases * 0.15)
    test_cases = n_cases - train_cases - val_cases

    assert count("train", "polyp") == train_cases * 33
    assert count("train", "non-polyp") == train_cases * 33
    assert count("val", "polyp") == val_cases * 33
    assert count("val", "non-polyp") == val_cases * 33
    assert count("test", "polyp") == test_cases * 33
    assert count("test", "non-polyp") == test_cases * 33

    for split in ["train", "val", "test"]:
        assert count(split, "polyp") == count(split, "non-polyp")

    # ensure case ids do not appear in multiple splits
    all_cases = {}
    for split in ["train", "val", "test"]:
        for cls in ["polyp", "non-polyp"]:
            for img in (out_dir / split / cls).glob("*.jpg"):
                cid = img.stem.split("_")[0]
                if cid in all_cases:
                    assert all_cases[cid] == split
                else:
                    all_cases[cid] = split


def test_prepare_sun_classification_morphology_stratified(tmp_path):
    paris = ["Is"] * 7 + ["IIa"] * 7
    canonical = _create_canonical_dataset(
        tmp_path / "canonical_morph", n_pos=14, n_neg_cases=14, paris_classes=paris
    )
    out_dir = tmp_path / "out_morph"
    prepare_sun_classification(
        canonical, out_dir, seed=0, stratify_morphology=True
    )

    def case_ids(split: str) -> set[str]:
        files = (out_dir / split / "polyp").glob("*.jpg")
        return {img.stem.split("_")[0] for img in files}

    polypoid_ids = {f"case{i:03d}" for i in range(1, 8)}
    flat_ids = {f"case{i:03d}" for i in range(8, 15)}

    train_ids = case_ids("train")
    val_ids = case_ids("val")
    test_ids = case_ids("test")

    assert len(train_ids & polypoid_ids) == 4
    assert len(train_ids & flat_ids) == 4
    assert len(val_ids & polypoid_ids) == 1
    assert len(val_ids & flat_ids) == 1
    assert len(test_ids & polypoid_ids) == 2
    assert len(test_ids & flat_ids) == 2

    def count(split: str, cls: str) -> int:
        return len(list((out_dir / split / cls).glob("*.jpg")))

    for split in ["train", "val", "test"]:
        assert count(split, "polyp") == count(split, "non-polyp")
