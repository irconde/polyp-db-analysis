"""Cropping helpers tailored for PolypGen negative sequence frames.

The negative sequences bundled with PolypGen include thick black borders and
letterboxed viewports that differ from the standard dataset imagery.  The
generic cropping utilities designed for annotated polyp frames struggle on
these negatives, frequently trimming away useful content or failing to remove
padding.  This module implements heuristics tuned specifically for those
frames.  Two complementary strategies are provided:

* ``border`` mode removes predominantly dark edge strips and then tightens the
  bounding box around the remaining non-dark pixels.  It is effective on
  frames that primarily consist of black padding around a central view.
* ``viewport`` mode identifies the main viewport by locating the largest band
  of columns that contain enough non-dark pixels.  The viewport is then cropped
  tightly to remove residual padding.

The :func:`crop_polypgen_negative` function exposes these strategies with an
``auto`` mode that chooses between them based on simple heuristics.  A PIL
wrapper is also provided for convenience when working with :class:`~PIL.Image`
objects.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _to_luma(arr: np.ndarray) -> np.ndarray:
    """Return the luma channel of ``arr`` as ``float32``."""

    if arr.ndim == 3:
        return (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        ).astype(np.float32)
    return arr.astype(np.float32)


def _tight_bbox(arr: np.ndarray, tol: int = 8) -> Tuple[int, int, int, int]:
    """Return the bounding box of pixels whose luma exceeds ``tol``."""

    y = _to_luma(arr)
    mask = y > tol
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        h, w = arr.shape[:2]
        return 0, h - 1, 0, w - 1
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _trim_edge_strips(
    arr: np.ndarray,
    *,
    tol: int = 40,
    edge_nonblack_max: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Trim predominantly dark rows and columns from the image edges."""

    y = _to_luma(arr)
    h, w = y.shape[:2]
    nonblack = y > tol

    left = 0
    while left < w - 1 and nonblack[:, left].mean() <= edge_nonblack_max:
        left += 1
    right = w - 1
    while right > left and nonblack[:, right].mean() <= edge_nonblack_max:
        right -= 1

    top = 0
    while top < h - 1 and nonblack[top, left : right + 1].mean() <= edge_nonblack_max:
        top += 1
    bottom = h - 1
    while bottom > top and nonblack[bottom, left : right + 1].mean() <= edge_nonblack_max:
        bottom -= 1

    return top, bottom, left, right


def _crop_black_borders(
    arr: np.ndarray,
    *,
    tol_edge: int = 40,
    edge_nonblack_max: float = 0.05,
    tol_tight: int = 8,
) -> np.ndarray:
    """Remove dark padding surrounding ``arr``."""

    top, bottom, left, right = _trim_edge_strips(
        arr, tol=tol_edge, edge_nonblack_max=edge_nonblack_max
    )
    cropped = arr[top : bottom + 1, left : right + 1]
    t2, b2, l2, r2 = _tight_bbox(cropped, tol=tol_tight)
    return cropped[t2 : b2 + 1, l2 : r2 + 1].copy()


def _find_largest_true_run(good: np.ndarray) -> Tuple[int, int]:
    """Locate the largest contiguous ``True`` run within ``good``."""

    max_len = 0
    best = (0, len(good) - 1)
    i = 0
    n = len(good)
    while i < n:
        if good[i]:
            j = i
            while j < n and good[j]:
                j += 1
            if (j - i) > max_len:
                max_len = j - i
                best = (i, j - 1)
            i = j
        else:
            i += 1
    return best


def _crop_main_viewport(
    arr: np.ndarray,
    *,
    edge_tol: int = 40,
    col_frac_thr: float = 0.30,
    tight_tol: int = 8,
) -> np.ndarray:
    """Crop around the main viewport by analysing non-dark column density."""

    y = _to_luma(arr)
    nonblack = y > edge_tol
    col_frac = nonblack.mean(axis=0)
    good = col_frac >= col_frac_thr
    left, right = _find_largest_true_run(good)
    sub = arr[:, left : right + 1]
    t, b, l2, r2 = _tight_bbox(sub, tol=tight_tol)
    return sub[t : b + 1, l2 : r2 + 1].copy()


def _decide_mode_auto(
    arr: np.ndarray,
    *,
    edge_tol: int = 40,
    col_frac_thr: float = 0.30,
) -> str:
    """Heuristically choose between ``viewport`` and ``border`` cropping."""

    y = _to_luma(arr)
    nonblack = y > edge_tol
    col_frac = nonblack.mean(axis=0)
    good = col_frac >= col_frac_thr
    left, _ = _find_largest_true_run(good)
    h, w = nonblack.shape
    left_band_width = max(1, int(0.2 * w))
    left_band_mean = col_frac[:left_band_width].mean()
    if left > int(0.2 * w) or left_band_mean <= 0.10:
        return "viewport"
    return "border"


def crop_polypgen_negative(
    arr: np.ndarray,
    *,
    mode: str = "auto",
    edge_tol: int = 40,
    edge_nonblack_max: float = 0.05,
    col_frac_thr: float = 0.30,
    tight_tol: int = 8,
) -> np.ndarray:
    """Crop ``arr`` using heuristics tuned for PolypGen negative frames."""

    if arr.ndim < 2:
        raise ValueError("Input array must have at least two dimensions")

    actual_mode = mode
    if mode == "auto":
        actual_mode = _decide_mode_auto(
            arr, edge_tol=edge_tol, col_frac_thr=col_frac_thr
        )

    if actual_mode == "viewport":
        return _crop_main_viewport(
            arr, edge_tol=edge_tol, col_frac_thr=col_frac_thr, tight_tol=tight_tol
        )
    if actual_mode != "border":
        raise ValueError(f"Unsupported cropping mode: {mode}")
    return _crop_black_borders(
        arr, tol_edge=edge_tol, edge_nonblack_max=edge_nonblack_max, tol_tight=tight_tol
    )


def crop_polypgen_negative_image(
    img: Image.Image,
    *,
    mode: str = "auto",
    edge_tol: int = 40,
    edge_nonblack_max: float = 0.05,
    col_frac_thr: float = 0.30,
    tight_tol: int = 8,
) -> Image.Image:
    """Return a cropped copy of ``img`` suitable for PolypGen negatives."""

    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL Image")
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    cropped = crop_polypgen_negative(
        arr,
        mode=mode,
        edge_tol=edge_tol,
        edge_nonblack_max=edge_nonblack_max,
        col_frac_thr=col_frac_thr,
        tight_tol=tight_tol,
    )
    return Image.fromarray(cropped)


__all__ = [
    "crop_polypgen_negative",
    "crop_polypgen_negative_image",
]

