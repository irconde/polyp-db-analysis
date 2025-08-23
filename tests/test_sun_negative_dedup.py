import shutil
from pathlib import Path
import logging

from PIL import Image

from polypdb.merge import copy_and_rename_negative_images


def test_negative_copy_deduplicates(tmp_path, caplog):
    src_dir = tmp_path / "sun_negatives" / "clip1"
    src_dir.mkdir(parents=True)
    img1 = src_dir / "a.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(img1)
    img2 = src_dir / "b.png"
    shutil.copy(img1, img2)

    dest_dir = tmp_path / "canonical" / "images"
    with caplog.at_level(logging.INFO):
        records, original = copy_and_rename_negative_images(
            str(src_dir.parent), str(dest_dir), dataset_name="sun"
        )

    # destination should only have one image
    assert len(list(Path(dest_dir).glob("*.png"))) == 1
    assert len(records) == 1
    assert original == 2
    assert "Found 2 negative images before dropping duplicates" in caplog.text
    assert (src_dir.parent / "neg_hash_cache.csv").exists()

    # originals remain untouched
    assert img1.exists()
    assert img2.exists()

