import numpy as np

from polypdb.polypgen_viewport import mask_small_viewport


def test_polypgen_viewport_masks_panel():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    img[60:100, 0:40] = (188, 229, 30)
    masked, box = mask_small_viewport(img)
    assert box is not None
    x1, y1, x2, y2 = box
    assert np.all(masked[y1:y2, x1:x2] == 50)


def test_polypgen_viewport_no_panel():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    masked, box = mask_small_viewport(img)
    assert box is None
    assert np.array_equal(masked, img)
