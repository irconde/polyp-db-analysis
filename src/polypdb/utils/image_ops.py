"""Image resizing and cropping helpers.

This module contains image related utility functions that were previously
located in :mod:`polypdb.utils`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import tifffile
from PIL import Image, ImageDraw

try:  # Optional dependency
    import cv2  # type: ignore

    if not hasattr(cv2, "imread"):
        cv2 = None  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    cv2 = None  # type: ignore


logger = logging.getLogger(__name__)


def _save_image(img, path):
    """Save ``img`` to ``path`` with TIFF fallback using ``tifffile``."""

    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tif", ".tiff"}:
        tifffile.imwrite(path, np.array(img))
    else:
        with open(path, "wb") as f:
            img.save(f)


def get_image_size(image_path):
    """Return the width and height of an image.

    The image is opened using :mod:`PIL.Image` instead of OpenCV.

    Args:
        image_path: Path to the image.

    Returns:
        Tuple ``(width, height)`` representing the size of the image.
    """

    with Image.open(image_path) as img:
        return img.width, img.height


def crop_image_padding(
    img_path,
    cropped_img_path,
    mask_path=None,
    cropped_mask_path=None,
    padding_color=(15, 15, 15),
    threshold=20,
):
    """Crop an image and optional mask, stripping away padding pixels.

    ``img_path`` is validated before attempting to open it.  Passing a non
    existing path (or an invalid file descriptor) previously resulted in a
    rather opaque ``OSError: [Errno 9] Bad file descriptor`` bubbling up from
    :func:`PIL.Image.open`.  By explicitly checking for path existence we can
    raise a clearer ``FileNotFoundError`` and avoid confusing descriptor
    errors.

    Parameters
    ----------
    img_path:
        Location of the input image to be cropped.
    cropped_img_path:
        Output location where the cropped image will be written.
    mask_path:
        Optional path to a segmentation mask corresponding to ``img_path``.
    cropped_mask_path:
        Destination for the cropped mask when ``mask_path`` is provided.
    padding_color:
        RGB colour considered as padding and removed from the image. Defaults
        to ``(15, 15, 15)``.
    threshold:
        Permissible channel deviation from ``padding_color`` to still be
        regarded as padding. Defaults to ``20``.
    """

    img_path = os.fspath(img_path)
    cropped_img_path = os.fspath(cropped_img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    with Image.open(img_path) as img:
        # Convert to RGB to consistently evaluate padding colour even when the
        # source image has an alpha channel or a single channel.  The original
        # ``img`` object is preserved so that the cropped output retains its
        # initial mode.
        rgb_img = img.convert("RGB")
        pixels = rgb_img.load()

        def is_close_to_color(pixel, color, threshold):
            return all(abs(int(p) - c) <= threshold for p, c in zip(pixel, color))

        # Determine the number of padding columns on each side
        left_padding = 0
        for x in range(rgb_img.width):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for y in range(rgb_img.height)
            ):
                left_padding += 1
            else:
                break

        right_padding = 0
        for x in range(rgb_img.width - 1, -1, -1):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for y in range(rgb_img.height)
            ):
                right_padding += 1
            else:
                break

        # Determine the number of padding rows on each side
        top_padding = 0
        for y in range(rgb_img.height):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for x in range(rgb_img.width)
            ):
                top_padding += 1
            else:
                break

        bottom_padding = 0
        for y in range(rgb_img.height - 1, -1, -1):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for x in range(rgb_img.width)
            ):
                bottom_padding += 1
            else:
                break

        if (
            left_padding + right_padding >= rgb_img.width
            or top_padding + bottom_padding >= rgb_img.height
        ):
            left, upper, right, lower = 0, 0, img.width, img.height
        else:
            left = left_padding
            upper = top_padding
            right = rgb_img.width - right_padding
            lower = rgb_img.height - bottom_padding

        # Crop the image to the bounding box to remove padding
        cropped_img = img.crop((left, upper, right, lower))
        _save_image(cropped_img, cropped_img_path)

    if mask_path is not None:
        mask_path = os.fspath(mask_path)
        cropped_mask_path = os.fspath(cropped_mask_path)
        if os.path.exists(mask_path):
            with Image.open(mask_path) as mask_img:
                # Crop the mask to the same bounding box
                cropped_mask_img = mask_img.crop((left, upper, right, lower))
                _save_image(cropped_mask_img, cropped_mask_path)


def crop_black_padding_band(
    img: np.ndarray,
    *,
    dark_threshold: int = 22,
    band: Tuple[float, float] = (0.18, 0.82),
    min_ratio: float = 0.50,
    dyn_percentile: float = 12.0,
    margin: int = 4,
    erode_px: int = 3,
    smooth_k: int = 11,
) -> Tuple[np.ndarray, int, int]:
    """Crop dark padding from ``img`` while ignoring on-screen overlays.

    The algorithm projects non-dark pixel densities onto the axes within a
    central band and determines dynamic thresholds that adapt to varying border
    sizes. It is resilient to text or logo overlays near the image edges by
    restricting the measurement to the specified ``band`` and smoothing the
    projections before thresholding.

    Returns the cropped image along with the x/y offset of the cropped region.
    """

    if cv2 is None:  # pragma: no cover - handled in tests
        raise ImportError("OpenCV is required for crop_black_padding_band")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = (gray > dark_threshold).astype(np.uint8) * 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask = cv2.erode(mask, k, iterations=1)

    y_lo, y_hi = int(band[0] * h), int(band[1] * h)
    x_lo, x_hi = int(band[0] * w), int(band[1] * w)

    col_counts = mask[y_lo:y_hi, :].sum(axis=0) // 255
    row_counts = mask[:, x_lo:x_hi].sum(axis=1) // 255

    def _smooth(arr: np.ndarray, k: int = smooth_k) -> np.ndarray:
        k = max(1, int(k))
        ker = np.ones(k, dtype=np.float32) / k
        return np.convolve(arr.astype(np.float32), ker, mode="same")

    col_sm = _smooth(col_counts)
    row_sm = _smooth(row_counts)

    col_thresh = max(min_ratio * (y_hi - y_lo), np.percentile(col_sm, dyn_percentile))
    row_thresh = max(min_ratio * (x_hi - x_lo), np.percentile(row_sm, dyn_percentile))

    x1 = 0
    for i in range(w):
        if col_sm[i] >= col_thresh:
            x1 = max(0, i - margin)
            break
    x2 = w
    for i in range(w - 1, -1, -1):
        if col_sm[i] >= col_thresh:
            x2 = min(w, i + margin)
            break

    y1 = 0
    for j in range(h):
        if row_sm[j] >= row_thresh:
            y1 = max(0, j - margin)
            break
    y2 = h
    for j in range(h - 1, -1, -1):
        if row_sm[j] >= row_thresh:
            y2 = min(h, j + margin)
            break

    if x2 <= x1 + 10 or y2 <= y1 + 10:
        return img, 0, 0

    return img[y1:y2, x1:x2].copy(), x1, y1


def resize_and_maintain_aspect(img_path, output_path, target_size=(854, 480), fill_color=(0, 0, 0)):
    """
    Resize an image to a target size maintaining the aspect ratio.
    If the image is smaller than the target, pad it with a specified color.
    :param img_path: Path to the input image.
    :param output_path: Path to the output image.
    :param target_size: Tuple (width, height) of the target size.
    :param fill_color: Tuple (R, G, B) of the color to fill the padded area.
    :return: None. The image is saved to the output path.
    """

    with Image.open(img_path) as img:
        # Maintain aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # If the image is smaller than the target, pad it
        if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
            new_img = Image.new("RGB", target_size, fill_color)
            new_img.paste(
                img,
                ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2),
            )
        else:
            new_img = img

        new_img.save(output_path)


def resize_with_padding(img, target_size=(224, 224), fill_color=(0, 0, 0)):
    """Resize ``img`` to ``target_size`` maintaining aspect ratio by padding."""

    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, fill_color)
    new_img.paste(
        img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
    )
    return new_img


def normalize_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize ``img`` using per-channel ``mean`` and ``std``."""

    arr = np.asarray(img).astype(np.float32) / 255.0
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)
    return (arr - mean_arr) / std_arr


def adjust_bounding_box(original_box, original_size, resized_size, padding):
    """Adjust bounding box coordinates for resized images with padding."""

    scale_x = resized_size[0] / original_size[0]
    scale_y = resized_size[1] / original_size[1]

    # Scale the original bounding box coordinates
    x1, y1, x2, y2 = original_box
    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    # Adjust for padding
    x1_adjusted = x1_scaled + padding[0]
    y1_adjusted = y1_scaled + padding[1]
    x2_adjusted = x2_scaled + padding[0]
    y2_adjusted = y2_scaled + padding[1]

    return x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted


def draw_bounding_boxes(bounding_boxes, image_path, save_path):
    """Overlay ``bounding_boxes`` on ``image_path`` and save to ``save_path``."""

    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        coordinates = [
            int(coord.strip()) for coord in bounding_boxes.split(",") if coord.strip()
        ]
        draw.rectangle(coordinates, outline="red", width=2)

        img.save(save_path)


__all__ = [
    "adjust_bounding_box",
    "crop_black_padding_band",
    "crop_image_padding",
    "draw_bounding_boxes",
    "get_image_size",
    "normalize_image",
    "resize_and_maintain_aspect",
    "resize_with_padding",
]

