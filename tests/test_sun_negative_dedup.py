import shutil
from pathlib import Path

from PIL import Image

from polypdb.preprocess.sun import drop_duplicate_negative_images


def test_drop_duplicate_negative_images(tmp_path):
    src_dir = tmp_path / "sun_negatives"
    src_dir.mkdir()

    img1 = src_dir / "a.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(img1)

    img2 = src_dir / "b.png"
    shutil.copy(img1, img2)

    img3 = src_dir / "c.png"
    Image.new("RGB", (12, 12), color=(0, 255, 0)).save(img3)

    removed = drop_duplicate_negative_images(str(src_dir))

    remaining = sorted(p.name for p in Path(src_dir).iterdir() if p.suffix == ".png")
    assert removed == 1
    assert remaining == ["a.png", "c.png"]

