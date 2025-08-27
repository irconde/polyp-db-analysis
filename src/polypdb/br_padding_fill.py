"""Utilities to neutralize bottom-right padding artefacts.

This module provides a helper to repaint the padding region that appears in the
bottom-right corner of some Hyper-Kvasir frames.  The function detects the
padding area itself rather than the UI glyph contained within it and fills it
with its dominant colour so that subsequent cropping can remove it cleanly.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ---------------- core: raw rule + BR component ----------------


def raw_rule_mask(img: np.ndarray, thresh: int = 17) -> np.ndarray:
    """Return a 0/255 mask where at least two channels are ``<= thresh``."""
    b, g, r = cv2.split(img)
    vals = np.stack([b, g, r], axis=-1).astype(np.int16)
    m = (np.sum(vals <= thresh, axis=-1) >= 2).astype(np.uint8) * 255
    return m


def strict_br_component(mask_raw: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Connected component touching bottom-right but not top/left borders."""
    H, W = mask_raw.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_raw > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return np.zeros((H, W), np.uint8)

    brx, bry = W - 1, H - 1
    best_i, best_d = None, 1e9

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        touches_right = x + w >= W
        touches_bottom = y + h >= H
        touches_left = x == 0
        touches_top = y == 0
        if not (touches_right and touches_bottom):
            continue
        if touches_left or touches_top:
            continue

        x2, y2 = x + w - 1, y + h - 1
        dx = 0 if x <= brx <= x2 else min(abs(brx - x), abs(brx - x2))
        dy = 0 if y <= bry <= y2 else min(abs(bry - y), abs(bry - y2))
        d = (dx * dx + dy * dy) ** 0.5
        if d < best_d:
            best_i, best_d = i, d

    out = np.zeros((H, W), np.uint8)
    if best_i is not None:
        out[labels == best_i] = 255
    return out


# ---------------- helpers: convexity, expand, colour ----------------


def _expand(mask: np.ndarray, r: int, close_ksize: int = 3) -> np.ndarray:
    out = mask.copy()
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        out = cv2.dilate(out, k, 1)
    if close_ksize > 0:
        k = np.ones((close_ksize, close_ksize), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, 1)
    return out


def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def convex_deficits(mask: np.ndarray) -> Tuple[float, float]:
    """Return global and top-band convexity deficit ratios."""
    cnt = _largest_contour(mask)
    if cnt is None:
        return 1.0, 1.0
    hull = cv2.convexHull(cnt)
    A_mask = float(cv2.contourArea(cnt))
    A_hull = float(cv2.contourArea(hull))
    if A_hull <= 0:
        return 1.0, 1.0

    def_ratio = max(0.0, (A_hull - A_mask) / A_hull)

    x, y, w, h = cv2.boundingRect(cnt)
    band = np.zeros_like(mask, np.uint8)
    y2 = y + max(1, int(0.25 * h))
    cv2.rectangle(band, (x, y), (x + w - 1, y2), 255, -1)

    mh = np.zeros_like(mask, np.uint8)
    cv2.fillConvexPoly(mh, hull, 255)

    missing = ((mh > 0) & (mask == 0)).astype(np.uint8) * 255
    missing_top = cv2.bitwise_and(missing, band)

    A_miss_top = float((missing_top > 0).sum())
    A_band_hull = float(((mh > 0) & (band > 0)).sum())
    top_def_ratio = 0.0 if A_band_hull == 0 else A_miss_top / A_band_hull

    return def_ratio, top_def_ratio


def dominant_bgr_from_raw(img: np.ndarray, raw_mask_br: np.ndarray) -> Tuple[int, int, int]:
    """Median BGR over pixels in ``raw_mask_br``."""
    px = img[raw_mask_br > 0].reshape(-1, 3)
    if px.size == 0:
        return (0, 0, 0)
    med = np.median(px, axis=0).astype(np.uint8)
    return int(med[0]), int(med[1]), int(med[2])


# ---------------- decision + paint ----------------


def decide_and_build_mask(
    raw_br: np.ndarray,
    conc_thr: float = 0.05,
    top_thr: float = 0.12,
    close_ksize: int = 3,
    r_if_irregular: int = 5,
) -> Tuple[int, np.ndarray]:
    """Determine dilation radius and return final mask."""
    base = _expand(raw_br, 0, close_ksize)
    def_all, def_top = convex_deficits(base)
    if def_all <= conc_thr and def_top <= top_thr:
        return 0, base
    return r_if_irregular, _expand(raw_br, r_if_irregular, close_ksize)


def fill_br_padding(
    img: np.ndarray,
    thresh: int = 17,
    min_area: int = 50,
    conc_thr: float = 0.05,
    top_thr: float = 0.12,
    close_ksize: int = 3,
    r_if_irregular: int = 5,
) -> np.ndarray:
    """Fill the bottom-right padding region with its dominant colour."""
    raw = strict_br_component(raw_rule_mask(img, thresh), min_area=min_area)
    if raw.sum() == 0:
        return img

    _, final_mask = decide_and_build_mask(
        raw,
        conc_thr=conc_thr,
        top_thr=top_thr,
        close_ksize=close_ksize,
        r_if_irregular=r_if_irregular,
    )
    color = dominant_bgr_from_raw(img, raw)

    out = img.copy()
    out[final_mask > 0] = color
    return out
