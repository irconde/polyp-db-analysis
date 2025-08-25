import numpy as np
from PIL import Image

from polypdb.utils import create_text_mask, create_text_masks


class DummyOCR:
    def ocr(self, img_path, det=True, rec=False, cls=False):
        return [([[10, 10], [20, 10], [20, 20], [10, 20]], 0.99)]


def test_create_text_mask(tmp_path):
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"

    Image.new("RGB", (30, 30), color=(0, 0, 0)).save(img_path)
    create_text_mask(img_path, mask_path, ocr=DummyOCR())

    with Image.open(mask_path) as m:
        arr = np.array(m)
    assert arr[15, 15] == 255
    assert arr[0, 0] == 0


def test_create_text_masks(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    for name in ["a.png", "b.jpg"]:
        Image.new("RGB", (30, 30)).save(src / name)

    create_text_masks(src, dst, ocr=DummyOCR())

    assert (dst / "a_mask.png").exists()
    assert (dst / "b_mask.png").exists()
