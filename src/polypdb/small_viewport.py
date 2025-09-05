"""Detection and masking for small bottom-left viewports.

This module provides utilities to detect the small inset viewport that can
appear in the bottom-left corner of some endoscopy frames.  The detection uses
an initial pass for dark bezels followed by a colour-based fallback.  The
located region can then be filled with a colour matching the surrounding
padding so that the artefact blends into the background.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import typing as t

import cv2  # type: ignore
import numpy as np

from .utils import crop_black_padding_band


# --------------------------- utilities ---------------------------

def darkest_padding_color(img: np.ndarray, pct: float = 0.02) -> tuple[int, int, int]:
    """Return the median BGR value of the darkest ``pct`` pixels in ``img``."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1)
    k = max(1, int(len(gray) * pct))
    idx = np.argpartition(gray, k)[:k]
    med = np.median(img.reshape(-1, 3)[idx], axis=0)
    return int(med[0]), int(med[1]), int(med[2])


# -------------------- small viewport detector --------------------

def find_small_viewport_bbox(
    img: np.ndarray,
    *,
    roi_frac: tuple[float, float, float, float] = (0.0, 0.45, 0.70, 1.0),
    bezel_dark_hi: tuple[int, int, int] = (40, 40, 40),
    min_side_px: int = 40,
    min_rel_area: float = 0.003,
    max_rel_area: float = 0.35,
    aspect_lo: float = 0.30,
    aspect_hi: float = 2.00,
    pad_px: int = 6,
) -> t.Optional[tuple[int, int, int, int]]:
    """Locate the small bottom-left viewport within ``img``.

    The search is restricted to the region specified by ``roi_frac``.  The
    function first looks for a dark bezel and falls back to a colour-based
    heuristic targeting green or blue panels.
    """

    h, w = img.shape[:2]
    x0 = int(roi_frac[0] * w)
    y0 = int(roi_frac[1] * h)
    x1 = int(roi_frac[2] * w)
    y1 = int(roi_frac[3] * h)
    roi = img[y0:y1, x0:x1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- 1) dark bezel pass ---
    dark = cv2.inRange(roi, (0, 0, 0), bezel_dark_hi)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: t.Optional[tuple[int, int, int, int]] = None
    best_area = -1
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_side_px or bh < min_side_px:
            continue
        area_rect = bw * bh
        if not (min_rel_area * w * h <= area_rect <= max_rel_area * w * h):
            continue
        asp = bw / float(bh)
        if not (aspect_lo <= asp <= aspect_hi):
            continue
        fill = cv2.contourArea(c) / (area_rect + 1e-6)
        if not (0.03 <= fill <= 0.60):
            continue
        left_bias = (x + bw / 2) / (x1 - x0 + 1e-6) < 0.70
        bottom_bias = (y + bh / 2) / (y1 - y0 + 1e-6) > 0.20
        if not (left_bias and bottom_bias):
            continue
        if area_rect > best_area:
            best_area = area_rect
            best = (x0 + x, y0 + y, x0 + x + bw, y0 + y + bh)

    # --- 2) color fallback (green/blue panel) ---
    if best is None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        color = ((h_ch >= 35) & (h_ch <= 140) & (s_ch >= 40) & (v_ch > 40)).astype(np.uint8) * 255
        color = cv2.morphologyEx(color, cv2.MORPH_OPEN, kernel, iterations=1)
        color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel, iterations=1)
        cnts, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < min_side_px or bh < min_side_px:
                continue
            area_rect = bw * bh
            if not (min_rel_area * w * h <= area_rect <= max_rel_area * w * h):
                continue
            asp = bw / float(bh)
            left_bias = (x + bw / 2) / (x1 - x0 + 1e-6) < 0.70
            bottom_bias = (y + bh / 2) / (y1 - y0 + 1e-6) > 0.20
            if aspect_lo <= asp <= aspect_hi and left_bias and bottom_bias:
                best = (x0 + x, y0 + y, x0 + x + bw, y0 + y + bh)
                break

    if best is None:
        return None

    bx1, by1, bx2, by2 = best
    return (
        max(0, bx1 - pad_px),
        max(0, by1 - pad_px),
        min(w, bx2 + pad_px),
        min(h, by2 + pad_px),
    )


def mask_small_viewport(
    img: np.ndarray,
    *,
    fill_mode: str = "auto",
    fill_color: t.Optional[tuple[int, int, int]] = None,
    **find_kw: t.Any,
) -> tuple[np.ndarray, t.Optional[tuple[int, int, int, int]]]:
    """Detect and mask the small viewport in ``img``.

    Parameters
    ----------
    img:
        Input BGR image.
    fill_mode:
        "auto" uses :func:`darkest_padding_color` to determine the fill colour,
        while "black" fills the region with solid black.
    fill_color:
        Explicit BGR colour to use instead of the automatic choices.
    find_kw:
        Additional keyword arguments forwarded to
        :func:`find_small_viewport_bbox`.
    """

    box = find_small_viewport_bbox(img, **find_kw)
    if not box:
        return img, None
    x1, y1, x2, y2 = box
    if fill_mode == "black":
        color = (0, 0, 0) if fill_color is None else fill_color
    else:
        color = darkest_padding_color(img) if fill_color is None else fill_color
    out = img.copy()
    out[y1:y2, x1:x2] = color
    return out, box


# ------------------------------ CLI ------------------------------

def main() -> None:  # pragma: no cover - convenience wrapper
    parser = argparse.ArgumentParser(
        description="Mask bottom-left mini-viewport in endoscopy frames.",
    )
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("output", type=Path, help="Output file or directory")
    parser.add_argument("--precrop", action="store_true", help="Pre-crop outer black padding before detection")
    parser.add_argument("--fill", choices=["auto", "black"], default="auto", help="Filling mode")
    parser.add_argument("--debug-box", action="store_true", help="Draw bbox instead of filling (for QA)")
    parser.add_argument(
        "--roi", type=float, nargs=4, default=(0.0, 0.45, 0.70, 1.0),
        help="ROI fractions x0 y0 x1 y1 for bottom-left search",
    )
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    in_path, out_path = args.input, args.output
    if out_path.suffix.lower() not in exts:
        out_path.mkdir(parents=True, exist_ok=True)

    def process_one(p: Path) -> None:
        img = cv2.imread(str(p))
        if img is None:
            return
        work = crop_black_padding_band(img)[0] if args.precrop else img
        if args.debug_box:
            box = find_small_viewport_bbox(work, roi_frac=tuple(args.roi))
            vis = work.copy()
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            out_file = out_path if out_path.suffix.lower() in exts else out_path / f"{p.stem}_sv_bbox{p.suffix}"
            cv2.imwrite(str(out_file), vis)
        else:
            masked, _ = mask_small_viewport(work, fill_mode=args.fill, roi_frac=tuple(args.roi))
            out_file = out_path if out_path.suffix.lower() in exts else out_path / f"{p.stem}_sv_masked{p.suffix}"
            cv2.imwrite(str(out_file), masked)

    if in_path.is_file():
        process_one(in_path)
    else:
        for p in in_path.rglob("*"):
            if p.suffix.lower() in exts:
                process_one(p)


if __name__ == "__main__":  # pragma: no cover
    main()
