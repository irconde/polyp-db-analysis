import numpy as np
import pytest

from polypdb.utils.masking import draw_mask_contours


def test_draw_mask_contours_outline():
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    outlined = draw_mask_contours(img, mask, color=(0, 255, 0), thickness=1)
    assert tuple(outlined[2, 2]) == (0, 255, 0)
    assert tuple(outlined[5, 5]) == (0, 0, 0)
