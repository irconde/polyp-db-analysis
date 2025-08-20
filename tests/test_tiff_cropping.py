import numpy as np
from PIL import Image
import tifffile

from polypdb.utils import crop_image_padding


def test_crop_image_padding_tiff(tmp_path):
    # Create an image with a white square in the centre and black padding
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[2:8, 2:8] = 255
    img = Image.fromarray(arr)
    mask = Image.fromarray(arr[:, :, 0])

    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.tif"
    img.save(img_path)
    tifffile.imwrite(mask_path, np.array(mask))

    out_img = tmp_path / "cropped.png"
    out_mask = tmp_path / "cropped.tif"

    crop_image_padding(
        img_path,
        out_img,
        mask_path=mask_path,
        cropped_mask_path=out_mask,
        padding_color=(0, 0, 0),
        threshold=0,
    )

    assert out_mask.exists()
    cropped = tifffile.imread(out_mask)
    assert cropped.shape == (6, 6)


def test_crop_image_padding_rgba(tmp_path):
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[..., 3] = 255  # opaque alpha channel
    arr[2:8, 2:8, :3] = 255
    img = Image.fromarray(arr, "RGBA")
    mask = Image.fromarray(arr[:, :, 0])

    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.tif"
    img.save(img_path)
    tifffile.imwrite(mask_path, np.array(mask))

    out_img = tmp_path / "cropped.png"
    out_mask = tmp_path / "cropped.tif"

    crop_image_padding(
        img_path,
        out_img,
        mask_path=mask_path,
        cropped_mask_path=out_mask,
        padding_color=(0, 0, 0),
        threshold=0,
    )

    assert Image.open(out_img).size == (6, 6)
    cropped = tifffile.imread(out_mask)
    assert cropped.shape == (6, 6)

