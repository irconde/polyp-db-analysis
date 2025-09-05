"""Detection and masking for small bottom-left viewports.

This module provides utilities to detect the small inset viewport that can
appear in the bottom-left corner of some endoscopy frames.  The detector seeds
on characteristic green/blue pixels, bridges occlusions to form a candidate
region and validates it using geometric heuristics before masking the panel
with a colour matching the surrounding padding.
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


def rect_score(contour: np.ndarray) -> float:
    """Return how well ``contour`` fills its minimum-area rectangle."""
    area = cv2.contourArea(contour)
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return float(area / max(w * h, 1.0))


def vh_edge_ratio(patch_gray: np.ndarray) -> float:
    """Fraction of Canny edges with roughly vertical or horizontal orientation."""
    g = cv2.GaussianBlur(patch_gray, (3, 3), 0)
    edges = cv2.Canny(g, 100, 200)
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    ang = np.degrees(np.arctan2(gy, gx))
    ang = np.mod(ang, 180.0)
    a = ang[edges > 0]
    if a.size == 0:
        return 0.0
    return float(
        np.mean((np.abs(a - 0) <= 15) | (np.abs(a - 90) <= 15) | (np.abs(a - 180) <= 15))
    )


def gb_seed_mask(bgr: np.ndarray, s_min: int = 50, v_min: int = 50) -> np.ndarray:
    """Binary mask of green/blue panel pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    m1 = (h_ch >= 35) & (h_ch <= 140) & (s_ch >= s_min) & (v_ch >= v_min)
    return (m1.astype(np.uint8)) * 255


# -------------------- small viewport detector --------------------

def find_small_viewport_bbox(
    img: np.ndarray,
    *,
    roi_frac: tuple[float, float, float, float] = (0.0, 0.45, 0.72, 1.0),
    min_side_px: int = 40,
    min_rel_area: float = 0.003,
    max_rel_area: float = 0.35,
    aspect_lo: float = 0.35,
    aspect_hi: float = 2.2,
    rect_thr: float = 0.72,
    vh_thr: float = 0.55,
    seed_min_area_px: int = 1500,
    dilate_px: int = 18,
    pad_px: int = 6,
) -> t.Optional[tuple[int, int, int, int]]:
    """Locate the small bottom-left viewport within ``img``.

    The search is restricted to the region specified by ``roi_frac``.  Green/
    blue panel pixels are used as seeds which are grown to bridge occlusions.
    Candidate regions are validated using rectangle coverage and vertical /
    horizontal edge ratios before being returned.
    """

    h, w = img.shape[:2]
    x0, y0 = int(roi_frac[0] * w), int(roi_frac[1] * h)
    x1, y1 = int(roi_frac[2] * w), int(roi_frac[3] * h)
    roi = img[y0:y1, x0:x1]

    # 1) Seed on green/blue panel pixels
    seed = gb_seed_mask(roi)
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1
    )
    n_seed = int(seed.sum() // 255)
    if n_seed < seed_min_area_px:
        return None

    # 2) Grow to bridge tools/occlusions (convex hull + dilation)
    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    grown = np.zeros(seed.shape, np.uint8)
    cv2.drawContours(grown, [hull], -1, 255, -1)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    grown = cv2.dilate(grown, k, iterations=1)

    # 3) Candidate box from grown mask
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    area = bw * bh
    if bw < min_side_px or bh < min_side_px:
        return None
    if not (min_rel_area * w * h <= area <= max_rel_area * w * h):
        return None
    asp = bw / float(bh)
    if not (aspect_lo <= asp <= aspect_hi):
        return None

    # location bias (keep bottom-left)
    if (bx + bw / 2) / (x1 - x0 + 1e-6) >= 0.80:
        return None
    if (by + bh / 2) / (y1 - y0 + 1e-6) <= 0.18:
        return None

    # 4) Rectangle & edge validation
    rect = rect_score(max(cnts, key=cv2.contourArea))
    ring = roi[
        max(by - 6, 0) : min(by + bh + 6, roi.shape[0]),
        max(bx - 6, 0) : min(bx + bw + 6, roi.shape[1]),
    ]
    vh = vh_edge_ratio(cv2.cvtColor(ring, cv2.COLOR_BGR2GRAY))
    if rect < rect_thr or vh < vh_thr:
        return None

    # 5) Return expanded box in full-image coordinates
    x1b = max(0, x0 + bx - pad_px)
    y1b = max(0, y0 + by - pad_px)
    x2b = min(w, x0 + bx + bw + pad_px)
    y2b = min(h, y0 + by + bh + pad_px)
    return (x1b, y1b, x2b, y2b)


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


def find_small_viewport_bbox_permissive(
    img: np.ndarray,
    *,
    roi_frac: tuple[float, float, float, float] = (0.0, 0.45, 0.72, 1.0),
    seed_min_area_px: int = 800,
    dilate_px: int = 16,
    min_side_px: int = 40,
    min_rel_area: float = 0.002,
    max_rel_area: float = 0.35,
    aspect_lo: float = 0.30,
    aspect_hi: float = 2.8,
    center_x_max: float = 0.82,
    center_y_min: float = 0.18,
    fill_ratio_min: float = 0.12,
    side_touch_px: int = 8,
    side_touch_ratio: float = 0.05,
    pad_px: int = 6,
) -> t.Optional[tuple[int, int, int, int]]:
    """Locate a bottom-left viewport using permissive heuristics.

    The detector seeds on green/blue hues, bridges occlusions with a convex
    hull followed by dilation and applies a set of geometric checks that are
    tolerant to PolypGen frames where the panel may be partially obstructed.
    """

    h, w = img.shape[:2]
    x0, y0 = int(roi_frac[0] * w), int(roi_frac[1] * h)
    x1, y1 = int(roi_frac[2] * w), int(roi_frac[3] * h)
    roi = img[y0:y1, x0:x1]

    seed = gb_seed_mask(roi)
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1
    )
    if int(seed.sum() // 255) < seed_min_area_px:
        return None

    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    base = np.zeros(seed.shape, np.uint8)
    cv2.drawContours(base, [hull], -1, 255, -1)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    grown = cv2.dilate(base, k, 1)

    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gx, gy, gw, gh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    area = gw * gh
    if gw < min_side_px or gh < min_side_px:
        return None
    if not (min_rel_area * w * h <= area <= max_rel_area * w * h):
        return None
    asp = gw / float(gh)
    if not (aspect_lo <= asp <= aspect_hi):
        return None

    cx = (gx + gw / 2.0) / max(1e-6, (x1 - x0))
    cy = (gy + gh / 2.0) / max(1e-6, (y1 - y0))
    if cx >= center_x_max or cy <= center_y_min:
        return None

    support = (grown[gy:gy + gh, gx:gx + gw] // 255).astype(np.float32)
    if support.mean() < fill_ratio_min:
        return None

    h_sub, w_sub = support.shape
    left_strip = support[:, : min(side_touch_px, w_sub)]
    bot_strip = support[-min(side_touch_px, h_sub) :, :]
    if left_strip.mean() < side_touch_ratio and bot_strip.mean() < side_touch_ratio:
        return None

    x1b = max(0, x0 + gx - pad_px)
    y1b = max(0, y0 + gy - pad_px)
    x2b = min(w, x0 + gx + gw + pad_px)
    y2b = min(h, y0 + gy + gh + pad_px)
    return (x1b, y1b, x2b, y2b)


def mask_small_viewport_permissive(
    img: np.ndarray,
    **find_kw: t.Any,
) -> tuple[np.ndarray, t.Optional[tuple[int, int, int, int]]]:
    """Mask the viewport using :func:`find_small_viewport_bbox_permissive`.

    Parameters
    ----------
    img:
        Input BGR image.
    find_kw:
        Optional keyword overrides forwarded to
        :func:`find_small_viewport_bbox_permissive`.

    The region is filled using the dominant dark padding colour so that the
    result blends in with existing borders.
    """

    box = find_small_viewport_bbox_permissive(img, **find_kw)
    if not box:
        return img, None
    x1, y1, x2, y2 = box
    color = darkest_padding_color(img)
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
