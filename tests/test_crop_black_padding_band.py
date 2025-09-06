import numpy as np
import cv2

from polypdb.utils.image_ops import crop_black_padding_band


def test_crop_black_padding_band_ignores_overlays():
    # create central content and pad with black border
    core = np.full((80, 80, 3), 50, dtype=np.uint8)
    img = cv2.copyMakeBorder(core, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    # simulate overlay in padding region
    cv2.putText(img, "T", (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cropped, x_off, y_off = crop_black_padding_band(img)
    assert cropped.shape == (81, 81, 3)
    assert x_off == 9 and y_off == 9
