import numpy as np
from PIL import Image

import pytest

from polypdb.utils import create_text_mask, create_text_masks


class DummyOCR:
    def ocr(self, img, det=True, rec=False, cls=False):
        return [([[10, 10], [20, 10], [20, 20], [10, 20]], 0.99)]


@pytest.mark.parametrize("preprocess", [True, False])
def test_create_text_mask(tmp_path, preprocess):
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"

    Image.new("RGB", (30, 30), color=(0, 0, 0)).save(img_path)
    create_text_mask(img_path, mask_path, ocr=DummyOCR(), preprocess=preprocess)

    with Image.open(mask_path) as m:
        arr = np.array(m)
    assert arr[15, 15] == 255
    assert arr[0, 0] == 0


def test_create_text_mask_expansion(tmp_path):
    def fake_detect(img, ocr, scales=(1.0,), restrict_to_edges=True, suppress_glare=True):
        return [(10, 10, 20, 20)], None

    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (30, 30), color=(0, 0, 0)).save(img_path)
    from polypdb import utils

    utils.detect_text_boxes, orig = fake_detect, utils.detect_text_boxes
    try:
        create_text_mask(img_path, mask_path, ocr=DummyOCR(), preprocess=False, expand=3)
    finally:
        utils.detect_text_boxes = orig
    arr = np.array(Image.open(mask_path))
    assert arr[15, 7] == 255
    assert arr[15, 6] == 0


def test_create_text_mask_left_edge(tmp_path):
    def fake_detect(img, ocr, scales=(1.0,), restrict_to_edges=True, suppress_glare=True):
        return [(2, 10, 8, 20)], None

    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (30, 30), color=(0, 0, 0)).save(img_path)
    from polypdb import utils

    utils.detect_text_boxes, orig = fake_detect, utils.detect_text_boxes
    try:
        create_text_mask(img_path, mask_path, ocr=DummyOCR(), preprocess=False, expand=5)
    finally:
        utils.detect_text_boxes = orig
    arr = np.array(Image.open(mask_path))
    assert arr[15, 0] == 255


@pytest.mark.parametrize("preprocess", [True, False])
def test_create_text_masks(tmp_path, preprocess):
    src = tmp_path / f"src_{preprocess}"
    dst = tmp_path / f"dst_{preprocess}"
    src.mkdir()
    dst.mkdir()
    for name in ["a.png", "b.jpg"]:
        Image.new("RGB", (30, 30)).save(src / name)

    create_text_masks(src, dst, ocr=DummyOCR(), preprocess=preprocess)

    assert (dst / "a_mask.png").exists()
    assert (dst / "b_mask.png").exists()
