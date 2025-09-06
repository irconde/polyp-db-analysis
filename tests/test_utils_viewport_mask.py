import numpy as np

from polypdb.utils.masking import mask_viewport_auto_color_safe


def test_mask_viewport_applies():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[60:, :40] = (135, 162, 18)
    masked, color, applied = mask_viewport_auto_color_safe(image)
    assert applied
    assert (masked[99, 0] == 0).all()
    assert (masked[0, 99] == 255).all()


def test_mask_viewport_skips_when_color_differs():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[60:, :40] = (0, 0, 255)
    masked, color, applied = mask_viewport_auto_color_safe(image)
    assert not applied
    assert np.array_equal(masked, image)

