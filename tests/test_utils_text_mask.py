import numpy as np
from PIL import Image, ImageDraw

import pytest

from polypdb.utils.masking import create_text_mask, create_text_masks


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
    def fake_detect(
        img,
        ocr,
        scales=(1.0,),
        restrict_to_edges=True,
        suppress_glare=True,
        preprocess=True,
    ):
        return [(10, 10, 20, 20)], None

    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (30, 30), color=(0, 0, 0)).save(img_path)
    import polypdb.utils.masking as utils

    utils.detect_text_boxes, orig = fake_detect, utils.detect_text_boxes
    try:
        create_text_mask(
            img_path,
            mask_path,
            ocr=DummyOCR(),
            preprocess=False,
            expand=3,
            left_clamp=0,
        )
    finally:
        utils.detect_text_boxes = orig
    arr = np.array(Image.open(mask_path))
    assert arr[15, 7] == 255
    assert arr[15, 6] == 0


def test_create_text_mask_left_edge(tmp_path):
    def fake_detect(
        img,
        ocr,
        scales=(1.0,),
        restrict_to_edges=True,
        suppress_glare=True,
        preprocess=True,
    ):
        return [(20, 10, 24, 20)], None

    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (30, 30), color=(0, 0, 0)).save(img_path)
    import polypdb.utils.masking as utils

    utils.detect_text_boxes, orig = fake_detect, utils.detect_text_boxes
    try:
        create_text_mask(img_path, mask_path, ocr=DummyOCR(), preprocess=False, expand=5)
    finally:
        utils.detect_text_boxes = orig
    arr = np.array(Image.open(mask_path))
    assert arr[15, 0] == 255


def test_create_text_mask_ui_squares(tmp_path):
    pytest.importorskip("cv2")

    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"

    img = Image.new("RGB", (100, 20), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(6):
        x = 5 + i * 15
        draw.rectangle((x, 5, x + 10, 15), fill=(255, 255, 255))
    img.save(img_path)

    import polypdb.utils.masking as utils

    utils.detect_text_boxes, orig = (
        lambda img, ocr, **kwargs: ([], None),
        utils.detect_text_boxes,
    )
    try:
        create_text_mask(img_path, mask_path, ocr=DummyOCR(), preprocess=False)
    finally:
        utils.detect_text_boxes = orig

    arr = np.array(Image.open(mask_path))
    assert arr[10, 10] == 255
    assert arr[10, 99] == 0


def test_create_text_mask_debug(tmp_path):
    pytest.importorskip("cv2")
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    debug_path = tmp_path / "debug.png"

    Image.new("RGB", (30, 30), color=(0, 0, 0)).save(img_path)

    import polypdb.utils.masking as utils

    def fake_detect(img, ocr, **kwargs):
        return [(10, 10, 20, 20)], None

    def fake_ui_squares(img):
        m = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        m[5:15, 2:8] = 255
        return m
    utils.detect_text_boxes, orig_detect = fake_detect, utils.detect_text_boxes
    utils.detect_ui_squares, orig_ui = fake_ui_squares, utils.detect_ui_squares
    try:
        create_text_mask(
            img_path,
            mask_path,
            ocr=DummyOCR(),
            preprocess=False,
            expand=0,
            left_clamp=0,
            debug_path=debug_path,
        )
    finally:
        utils.detect_text_boxes = orig_detect
        utils.detect_ui_squares = orig_ui

    debug_img = np.array(Image.open(debug_path))
    assert (debug_img[10, 2] == np.array([0, 255, 0])).all()
    assert (debug_img[10, 15] == np.array([255, 0, 0])).all()


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


def test_create_text_masks_debug_dir(tmp_path):
    pytest.importorskip("cv2")
    src = tmp_path / "src"
    mask_dst = tmp_path / "masks"
    debug_dst = tmp_path / "debug"
    src.mkdir()
    mask_dst.mkdir()
    Image.new("RGB", (30, 30)).save(src / "a.png")

    import polypdb.utils.masking as utils

    utils.detect_text_boxes, orig = (
        lambda img, ocr, **kwargs: ([], None),
        utils.detect_text_boxes,
    )
    try:
        create_text_masks(
            src,
            mask_dst,
            ocr=DummyOCR(),
            preprocess=False,
            debug_dir=debug_dst,
        )
    finally:
        utils.detect_text_boxes = orig

    assert (mask_dst / "a_mask.png").exists()
    assert (debug_dst / "a.png").exists()


def test_create_text_masks_inpaint(tmp_path):
    src = tmp_path / "src"
    mask_dst = tmp_path / "masks"
    inpaint_dst = tmp_path / "inpaint"
    src.mkdir()
    mask_dst.mkdir()
    inpaint_dst.mkdir()

    Image.new("RGB", (30, 30)).save(src / "a.png")

    class DummyInpaint:
        def __call__(self, image, mask):
            return image

    create_text_masks(
        src,
        mask_dst,
        ocr=DummyOCR(),
        preprocess=False,
        inpaint_dir=inpaint_dst,
        inpaint_model=DummyInpaint(),
    )

    assert (mask_dst / "a_mask.png").exists()
    assert (inpaint_dst / "a.png").exists()
