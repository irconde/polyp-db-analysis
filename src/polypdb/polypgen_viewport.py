"""Robust detection and masking of PolypGen's small bottom-left viewport.

This module provides a colour-based detector for the small inset viewport
present in many PolypGen frames.  The detector anchors on characteristic
teal/blue colours and validates the candidate region with geometric heuristics
before masking it using the dominant dark padding colour.  It is an alternative
implementation to :mod:`polypdb.small_viewport` tuned specifically for the
PolypGen dataset.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np


# -------------------- utilities --------------------

def snap_into_dark_frame(
    img_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    max_snap: Optional[int] = None,
    v_dark: int = 42,
    min_dark_frac: float = 0.70,
) -> Tuple[int, int, int, int]:
    """Expand ``(x1, y1, x2, y2)`` into the surrounding dark padding.

    The search proceeds downwards and leftwards while the newly included pixels
    remain mostly dark.  This helps align the box with the outer padding frame
    when the detector initially locks onto the inner coloured panel.
    """

    h, w = img_bgr.shape[:2]
    if max_snap is None:
        max_snap = int(0.25 * max(h, w))
    V = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]

    steps = 0
    # Snap downwards
    while steps < max_snap:
        row = V[min(h - 1, y2) : min(h, y2 + 1), x1:x2]
        if row.size == 0 or (row < v_dark).mean() < min_dark_frac:
            break
        y2 += 1
        steps += 1

    steps = 0
    # Snap leftwards
    while steps < max_snap:
        col = V[y1:y2, max(0, x1 - 1) : x1]
        if col.size == 0 or (col < v_dark).mean() < min_dark_frac:
            break
        x1 -= 1
        steps += 1

    return x1, y1, x2, y2


def darkest_padding_color(img: np.ndarray, pct: float = 0.02) -> Tuple[int, int, int]:
    """Median colour of the darkest ``pct`` pixels in ``img``.

    The darkest padding surrounding the viewport is typically a solid colour;
    using the median of the darkest pixels yields a robust fill value.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1)
    k = max(1, int(len(gray) * pct))
    idx = np.argpartition(gray, k)[:k]
    return tuple(int(v) for v in np.median(img.reshape(-1, 3)[idx], axis=0))


def rgb_to_lab_triplets(rgb_list: list[Tuple[int, int, int]]) -> np.ndarray:
    """Convert a list of RGB colours to an ``(N, 3)`` array in Lab space."""

    arr = np.array(rgb_list, dtype=np.uint8).reshape(1, len(rgb_list), 3)
    arr_bgr = arr[..., ::-1]  # convert to BGR for OpenCV
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).reshape(len(rgb_list), 3)
    return lab.astype(np.float32)


def lab_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return CIE76 distances between ``a`` and each colour in ``b``.

    Parameters
    ----------
    a:
        ``(H, W, 3)`` Lab image.
    b:
        ``(N, 3)`` Lab colour prototypes.
    Returns
    -------
    ``(H, W, N)`` array of distances.
    """

    a_exp = a[:, :, None, :]
    b_exp = b[None, None, :, :]
    return np.linalg.norm(a_exp - b_exp, axis=3)


def edge_density(gray_roi: np.ndarray) -> float:
    e = cv2.Canny(gray_roi, 50, 150)
    return float(e.sum()) / (255.0 * e.size)


def rectangularity(bin_mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return cv2.contourArea(c) / float(w * h + 1e-6)


def border_contrast_ok(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    band: int = 8,
    min_abs_mean: float = 16.0,
    max_std: float = 48.0,
) -> bool:
    """Check that borders around ``(x1, y1, x2, y2)`` show sufficient contrast."""

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    if y1 - band < 0 or x2 + band >= w:
        return False
    inside_top = V[y1:y1 + band, x1:x2]
    outside_top = V[max(0, y1 - band) : y1, x1:x2]
    inside_r = V[y1:y2, max(0, x2 - band) : x2]
    outside_r = V[y1:y2, x2 : min(w, x2 + band)]
    dt = inside_top - outside_top
    dr = inside_r - outside_r
    m_t, s_t = float(np.mean(dt)), float(np.std(dt))
    m_r, s_r = float(np.mean(dr)), float(np.std(dr))
    return (
        abs(m_t) >= min_abs_mean
        and s_t <= max_std
        and abs(m_r) >= min_abs_mean
        and s_r <= max_std
    )


# Panel colour prototypes (RGB). Green/teal from examples; blue from overlays.
GREEN_PROTOS = [(30, 229, 188), (6, 175, 144), (14, 219, 178)]
BLUE_PROTOS = [(60, 102, 179), (52, 117, 216), (44, 134, 229)]
LAB_PROTOS = np.vstack([rgb_to_lab_triplets(GREEN_PROTOS), rgb_to_lab_triplets(BLUE_PROTOS)])


def grow_box(
    box: Tuple[int, int, int, int],
    shape: Tuple[int, int, int],
    *,
    pad_px: int = 8,
) -> Tuple[int, int, int, int]:
    """Expand ``box`` by ``pad_px`` in all directions respecting ``shape``."""

    x1, y1, x2, y2 = box
    h, w = shape[:2]
    return (
        max(0, x1 - pad_px),
        max(0, y1 - pad_px),
        min(w, x2 + pad_px),
        min(h, y2 + pad_px),
    )


def detect_small_viewport(
    img_bgr: np.ndarray,
    *,
    roi_w_frac: float = 0.62,
    roi_h_frac: float = 0.62,
    lab_thr: float = 34.0,
    sat_min: int = 80,
    val_min: int = 60,
    min_w_frac: float = 0.08,
    min_h_frac: float = 0.10,
    max_w_frac: float = 0.55,
    max_h_frac: float = 0.55,
    comp_dominance: float = 0.75,
    edge_tol_rel: float = 0.08,
    max_edge_density: float = 0.18,
    min_rectangularity: float = 0.68,
) -> Optional[Tuple[int, int, int, int]]:
    """Locate the small bottom-left viewport within ``img_bgr``.

    The detector is tuned for the PolypGen dataset and returns the bounding box
    as ``(x1, y1, x2, y2)`` in image coordinates when a viewport is found.
    """

    H, W = img_bgr.shape[:2]

    # 1) bottom-left ROI
    x0, y0 = 0, int(H * (1.0 - roi_h_frac))
    x1r, y1r = int(W * roi_w_frac), H
    ROI = img_bgr[y0:y1r, x0:x1r]

    # 2) colour gating in Lab + HSV
    lab = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    d = lab_distance(lab, LAB_PROTOS).min(axis=2)
    mask_color = (d <= lab_thr) & (S >= sat_min) & (V >= val_min)
    mask = (mask_color.astype(np.uint8) * 255)

    # 3) morphology, largest component
    kx = max(3, int(round(W * 0.02)))
    ky = max(3, int(round(H * 0.01)))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kx | 1, ky | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 4) size & anchoring
    if not (W * min_w_frac <= w <= W * max_w_frac):
        return None
    if not (H * min_h_frac <= h <= H * max_h_frac):
        return None

    X1, Y1, X2, Y2 = x0 + x, y0 + y, x0 + x + w, y0 + y + h
    X1, Y1, X2, Y2 = snap_into_dark_frame(img_bgr, X1, Y1, X2, Y2)

    tol = max(8, int(edge_tol_rel * min(H, W)))
    if not (X1 <= tol and (H - Y2) <= tol):
        return None

    # 5) Dominance & flatness computed on the component only
    comp_roi = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(comp_roi, [c], -1, 255, -1)
    comp_box = comp_roi[(Y1 - y0) : (Y2 - y0), (X1 - x0) : (X2 - x0)] > 0

    roi_box = img_bgr[Y1:Y2, X1:X2]
    lab_box = cv2.cvtColor(roi_box, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv_box = cv2.cvtColor(roi_box, cv2.COLOR_BGR2HSV)

    d_box = lab_distance(lab_box, LAB_PROTOS).min(axis=2)
    cm_box = (
        (d_box <= lab_thr)
        & (hsv_box[:, :, 1] >= sat_min)
        & (hsv_box[:, :, 2] >= val_min)
    )

    comp_pixels = cm_box[comp_box]
    if comp_pixels.size == 0 or float(comp_pixels.mean()) < comp_dominance:
        return None

    v_std = float(hsv_box[:, :, 2][cm_box & comp_box].std())
    if v_std > 30.0:
        return None

    sub = np.zeros((h, w), np.uint8)
    cv2.drawContours(sub, [c - [x, y]], -1, 255, -1)
    if rectangularity(sub) < min_rectangularity:
        return None

    return (X1, Y1, X2, Y2)


def mask_small_viewport(
    img_bgr: np.ndarray,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Mask the small viewport in ``img_bgr`` if detected."""

    box = detect_small_viewport(img_bgr)
    if box is None:
        return img_bgr.copy(), None
    pad_px = max(4, int(0.006 * max(img_bgr.shape[:2])))
    x1, y1, x2, y2 = grow_box(box, img_bgr.shape, pad_px=pad_px)
    out = img_bgr.copy()
    out[y1:y2, x1:x2] = darkest_padding_color(img_bgr)
    return out, box


# ---------- CLI ----------

def process_path(inp: str, out_dir: str, write_debug: bool = False) -> None:
    paths: list[str] = []
    if os.path.isdir(inp):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            paths.extend(glob.glob(os.path.join(inp, ext)))
    else:
        paths = [inp]

    os.makedirs(out_dir, exist_ok=True)

    for p in sorted(paths):
        img = cv2.imread(p)
        if img is None:
            print(f"[skip] cannot read: {p}")
            continue
        masked, box = mask_small_viewport(img)
        name, ext = os.path.splitext(os.path.basename(p))
        out_path = os.path.join(out_dir, f"{name}_masked{ext}")
        cv2.imwrite(out_path, masked)

        if write_debug:
            dbg = img.copy()
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 4)
            dbg_path = os.path.join(out_dir, f"{name}_bbox{ext}")
            cv2.imwrite(dbg_path, dbg)

        print(f"[ok] {p} -> {out_path}  bbox={box}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mask bottom-left mini-viewport in endoscopy frames."
    )
    ap.add_argument("--input", required=True, help="Image file or directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--debug", action="store_true", help="Write images with detection rectangles"
    )
    args = ap.parse_args()
    process_path(args.input, args.out, write_debug=args.debug)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
