import numpy as np
import pytest

from polypdb.utils import preprocess_for_text


def test_preprocess_for_text_shape():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:10, :] = 255  # half bright, half dark
    out = preprocess_for_text(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
