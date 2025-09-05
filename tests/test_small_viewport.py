import numpy as np

from polypdb.small_viewport import (
    mask_small_viewport,
    mask_small_viewport_permissive,
)


def test_mask_small_viewport_masks_panel():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    img[60:100, 5:55] = (0, 255, 0)
    masked, box = mask_small_viewport(img, dilate_px=5, max_rel_area=0.5)
    assert box is not None
    x1, y1, x2, y2 = box
    assert np.all(masked[y1:y2, x1:x2] == 50)


def test_mask_small_viewport_no_panel():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    masked, box = mask_small_viewport(img, dilate_px=5, max_rel_area=0.5)
    assert box is None
    assert np.array_equal(masked, img)


def test_mask_small_viewport_permissive_masks_panel():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    img[60:100, 0:50] = (0, 255, 0)
    masked, box = mask_small_viewport_permissive(img, max_rel_area=0.6)
    assert box is not None
    x1, y1, x2, y2 = box
    assert np.all(masked[y1:y2, x1:x2] == 50)


def test_mask_small_viewport_permissive_no_panel():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    masked, box = mask_small_viewport_permissive(img)
    assert box is None
    assert np.array_equal(masked, img)
