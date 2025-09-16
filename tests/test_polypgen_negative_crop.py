import numpy as np
from PIL import Image

from polypdb.polypgen_negative_crop import (
    crop_polypgen_negative,
    crop_polypgen_negative_image,
)


def test_crop_polypgen_negative_border_mode_removes_padding():
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[5:15, 5:15] = 200

    cropped = crop_polypgen_negative(arr, mode="border")

    assert cropped.shape == (10, 10, 3)
    assert cropped.mean() > 0


def test_crop_polypgen_negative_viewport_mode_selects_main_region():
    arr = np.zeros((30, 50, 3), dtype=np.uint8)
    arr[5:25, 30:48] = 255

    cropped = crop_polypgen_negative(arr, mode="viewport")

    assert cropped.shape == (20, 18, 3)
    assert cropped[..., 0].max() == 255


def test_crop_polypgen_negative_image_auto_mode_returns_smaller_image():
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[4:28, 40:62] = 180
    img = Image.fromarray(arr)

    cropped_img = crop_polypgen_negative_image(img, mode="auto")

    assert isinstance(cropped_img, Image.Image)
    assert cropped_img.size[0] < img.width
    assert cropped_img.size[1] <= img.height
