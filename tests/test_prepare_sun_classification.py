from pathlib import Path
import pandas as pd
from PIL import Image

from polypdb.sun_classification import prepare_sun_classification


def _create_canonical_dataset(root: Path, n_pos: int = 7, n_neg_cases: int = 7, frames_per_case: int = 33) -> Path:
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
    pd.DataFrame({"global_case_id": [f"P{i}" for i in range(n_pos)] + [neg_case_id]}).to_parquet(
        manifests_dir / "cases.parquet", index=False
    )
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
