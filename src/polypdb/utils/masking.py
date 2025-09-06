"""Mask generation and related helpers.

Functions in this module deal with generating masks for various artefacts and
text regions. They were previously part of :mod:`polypdb.utils`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple
import typing as t

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:  # Optional dependency
    import cv2  # type: ignore

    if not hasattr(cv2, "imread"):
        cv2 = None  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    cv2 = None  # type: ignore

from .image_ops import _save_image, crop_image_padding


logger = logging.getLogger(__name__)


def mask_corner_artifacts(
    img: Image.Image,
    width: int = 50,
    height: int = 50,
    auto_detect: bool = True,
    threshold: int = 10,
    max_colour_run: int = 25,
    dark_thresh: int = 40,
    max_auto_fraction: float = 0.15,
    fill_mode: str = "base",
):
    """Mask camera overlays or logos in the four image corners."""

    img = img.convert("RGB")
    pixels = img.load()
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    gray = cv2.cvtColor(np.array(img)[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    dark = gray < dark_thresh

    fixed_fill: Optional[Tuple[int, int, int]] = None
    if fill_mode.startswith("rgb:"):
        try:
            fixed_fill = tuple(int(c) for c in fill_mode.split("rgb:")[1].split(","))
            if len(fixed_fill) != 3:
                fixed_fill = None
        except Exception:  # pragma: no cover - robustness
            fixed_fill = None

    def _differs(p1, p2):
        return any(abs(int(a) - int(b)) > threshold for a, b in zip(p1[:3], p2[:3]))

    def _measure(x0: int, y0: int, x_step: int, y_step: int, max_w: int, max_h: int):
        base = pixels[x0, y0]
        # horizontal run along row y0
        run_w = 0
        x = x0
        miss = 0
        while 0 <= x < img_w and run_w < max_w:
            if _differs(pixels[x, y0], base) or not dark[y0, x]:
                miss += 1
                if miss > max_colour_run:
                    break
            else:
                miss = 0
            run_w += 1
            x += x_step
        if miss:
            run_w -= miss
            if miss > max_colour_run:
                run_w += 1
        run_w = max(0, run_w)

        # vertical run along column x0
        run_h = 0
        y = y0
        miss = 0
        while 0 <= y < img_h and run_h < max_h:
            if _differs(pixels[x0, y], base) or not dark[y, x0]:
                miss += 1
                if miss > max_colour_run:
                    break
            else:
                miss = 0
            run_h += 1
            y += y_step
        if miss:
            run_h -= miss
            if miss > max_colour_run:
                run_h += 1
        run_h = max(0, run_h)
        return run_w, run_h, base

    corners = [
        ("tl", 0, 0, 1, 1, lambda W, H: [(0, 0), (W, 0), (0, H)]),
        (
            "tr",
            img_w - 1,
            0,
            -1,
            1,
            lambda W, H: [(img_w, 0), (img_w - W, 0), (img_w, H)],
        ),
        (
            "bl",
            0,
            img_h - 1,
            1,
            -1,
            lambda W, H: [(0, img_h), (W, img_h), (0, img_h - H)],
        ),
        (
            "br",
            img_w - 1,
            img_h - 1,
            -1,
            -1,
            lambda W, H: [
                (img_w, img_h),
                (img_w - W, img_h),
                (img_w, img_h - H),
            ],
        ),
    ]

    for name, x0, y0, x_step, y_step, poly_fn in corners:
        if auto_detect:
            w, h, base = _measure(x0, y0, x_step, y_step, img_w, img_h)
            max_w = int(img_w * max_auto_fraction)
            max_h = int(img_h * max_auto_fraction)
            w = min(w, max_w)
            h = min(h, max_h)
        else:
            w, h, base = width, height, pixels[x0, y0]
        if w <= 0 or h <= 0:
            continue

        x1 = x0 + x_step * (w - 1)
        y1 = y0 + y_step * (h - 1)
        color = tuple(int(c) for c in (fixed_fill if fixed_fill is not None else base[:3]))

        if (
            0 <= x1 < img_w
            and 0 <= y1 < img_h
            and not _differs(pixels[x1, y1], base)
            and dark[y1, x1]
        ):
            if name == "tl":
                rect = [0, 0, w - 1, h - 1]
            elif name == "tr":
                rect = [img_w - w, 0, img_w - 1, h - 1]
            elif name == "bl":
                rect = [0, img_h - h, w - 1, img_h - 1]
            else:  # br
                rect = [img_w - w, img_h - h, img_w - 1, img_h - 1]
            draw.rectangle(rect, fill=color)
        else:
            draw.polygon(poly_fn(w, h), fill=color)

    return img


def crop_and_mask_corner_artifacts(
    src_dir,
    dest_dir,
    padding_color=(15, 15, 15),
    padding_threshold=20,
    threshold=10,
    dark_thresh=40,
    max_auto_fraction=0.15,
):
    """Mask corner overlays then crop padding using auto detection."""

    src_dir = os.fspath(src_dir)
    dest_dir = os.fspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    for name in os.listdir(src_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            continue
        in_path = os.path.join(src_dir, name)
        out_path = os.path.join(dest_dir, name)
        with Image.open(in_path) as img:
            masked = mask_corner_artifacts(
                img,
                auto_detect=True,
                threshold=threshold,
                dark_thresh=dark_thresh,
                max_auto_fraction=max_auto_fraction,
            )
            _save_image(masked, out_path)
        crop_image_padding(
            out_path,
            out_path,
            padding_color=padding_color,
            threshold=padding_threshold,
        )


def merge_rects(rects, gap=18, y_overlap=0.45, pad=2, H=None, W=None):
    """Merge nearby rectangles into larger boxes."""

    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    merged: t.List[t.Tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in rects:
        placed = False
        for i, (X0, Y0, X1, Y1) in enumerate(merged):
            ov = max(0, min(y1, Y1) - max(y0, Y0))
            if ov / max(1, min(y1 - y0, Y1 - Y0)) >= y_overlap and max(
                0, max(X0, x0) - min(X1, x1)
            ) <= gap:
                merged[i] = (min(X0, x0), min(Y0, y0), max(X1, x1), max(Y1, y1))
                placed = True
                break
        if not placed:
            merged.append((x0, y0, x1, y1))

    out = []
    for x0, y0, x1, y1 in merged:
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad
        if H is not None and W is not None:
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(W - 1, x1)
            y1 = min(H - 1, y1)
        out.append((x0, y0, x1, y1))
    return out


def preprocess_for_text(img: np.ndarray) -> np.ndarray:
    """Preprocess image for text detection when OpenCV is available."""

    if cv2 is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.equalizeHist(gray)
    norm = cv2.bitwise_not(norm)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def detect_text_boxes(
    img,
    ocr,
    scales=(1.0, 1.5, 2.0),
    restrict_to_edges=False,
    suppress_glare=False,
    preprocess=False,
):
    """Detect and merge rectangular text regions for ``img`` using ``ocr``."""

    H0, W0 = img.shape[:2]
    best_area, best_rects = -1, []

    for s in scales:
        if s != 1.0:
            if cv2 is None:
                continue
            resized = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        else:
            resized = img

        det_img = resized.copy()
        if cv2 is not None and suppress_glare:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            glare = ((hsv[..., 2] >= 242) & (hsv[..., 1] <= 60)).astype(np.uint8) * 255
            glare = cv2.dilate(
                glare, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1
            )
            det_img[glare > 0] = 0

        if preprocess:
            det_img = preprocess_for_text(det_img)

        res = ocr.ocr(det_img, cls=False)
        rects = []
        pages = (
            res
            if res and isinstance(res[0], list) and isinstance(res[0][0], (list, tuple))
            else [res]
        )
        for page in pages:
            if not page:
                continue
            for det in page:
                if not det or det[0] is None:
                    continue
                P = np.asarray(det[0], dtype=np.float32)
                x0, y0 = P.min(axis=0)
                x1, y1 = P.max(axis=0)
                w, h = x1 - x0, y1 - y0
                if w * h < 100 or h > 0.5 * resized.shape[0]:
                    continue
                if restrict_to_edges:
                    if not (
                        x0 < 0.45 * resized.shape[1]
                        or x1 > 0.55 * resized.shape[1]
                        or y0 < 0.25 * resized.shape[0]
                    ):
                        continue
                rects.append((int(x0 / s), int(y0 / s), int(x1 / s), int(y1 / s)))

        rects = merge_rects(rects, H=H0, W=W0)
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
        if area > best_area:
            best_area, best_rects = area, rects

    if best_rects:
        xmin = min(r[0] for r in best_rects)
        ymin = min(r[1] for r in best_rects)
        xmax = max(r[2] for r in best_rects)
        ymax = max(r[3] for r in best_rects)
        union = (xmin, ymin, xmax, ymax)
    else:
        union = None

    return best_rects, union


def detect_ui_square_row(
    bgr: np.ndarray,
    expected_count_range: t.Tuple[int, int] = (5, 9),
    search_band_ratio: float = 0.35,
    bright_pct: float = 96.0,
    min_area_ratio: float = 2e-5,
    max_area_ratio: float = 2e-3,
    fill_ratio_min: float = 0.5,
    square_ar_tol: float = 0.6,
    mean_int_min: int = 170,
    dilate_frac_of_box: float = 0.15,
) -> t.Optional[t.Dict[str, t.Any]]:
    """Detect the horizontal row of small UI squares typically on the left side."""

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    thr_val = max(200, int(np.percentile(g, bright_pct)))
    thr_val = min(254, thr_val)
    _, bin_hi = cv2.threshold(g, thr_val, 255, cv2.THRESH_BINARY)

    left_w = int(search_band_ratio * w)
    if w < 200:
        left_w = w
    left_band = bin_hi[:, :left_w]

    min_area = max(1, int((w * h) * min_area_ratio))
    max_area = max(150, int((w * h) * max_area_ratio))

    contours, _ = cv2.findContours(
        left_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    squares: t.List[t.Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < min_area or area > max_area:
            continue
        ar = ww / float(hh) if hh else 0.0
        if ar < square_ar_tol or ar > (1.0 / square_ar_tol):
            continue
        cnt_area = cv2.contourArea(c)
        if cnt_area / max(area, 1) < fill_ratio_min:
            continue
        mask = np.zeros_like(left_band, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_int = cv2.mean(g[:, :left_w], mask=mask)[0]
        if mean_int < mean_int_min:
            continue
        squares.append((x, y, ww, hh))

    if not squares:
        return None

    squares = sorted(squares, key=lambda b: (b[1], b[0]))

    rows: t.List[t.List[t.Tuple[int, int, int, int]]] = []
    for s in squares:
        placed = False
        for row in rows:
            yh = np.mean([r[3] for r in row])
            if abs(s[1] - np.mean([r[1] for r in row])) <= max(6, 0.8 * yh):
                row.append(s)
                placed = True
                break
        if not placed:
            rows.append([s])

    candidates = []
    max_gap_px = int(0.015 * w)
    for row in rows:
        row = sorted(row, key=lambda b: b[0])
        n = len(row)
        if expected_count_range[0] <= n <= expected_count_range[1]:
            xs = [b[0] for b in row]
            gaps = np.diff(xs)
            if len(gaps) == 0:
                continue
            mean_w = np.mean([b[2] for b in row])
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            if (
                std_gap <= max(5.0, 0.7 * mean_w)
                and mean_gap < 3 * mean_w
                and mean_gap < max_gap_px
            ):
                candidates.append(row)

    if candidates:
        row = sorted(
            min(candidates, key=lambda r: np.mean([b[0] for b in r])),
            key=lambda b: b[0],
        )
    else:
        row = sorted(
            sorted(rows, key=lambda r: (-len(r), np.mean([b[0] for b in r])))[0],
            key=lambda b: b[0],
        )

    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, ww, hh) in row:
        cv2.rectangle(mask, (x, y), (x + ww, y + hh), 255, -1)

    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bbox: t.Tuple[int, int, int, int] = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    k = max(1, int(round(dilate_frac_of_box * np.mean([b[2] for b in row]))))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    return {"mask": mask, "row_boxes": row, "bbox": bbox}


def detect_ui_squares(bgr: np.ndarray) -> t.Optional[np.ndarray]:
    """Detect the left-side row of UI squares and return a mask covering them."""

    det = detect_ui_square_row(bgr)
    if det is None:
        return None
    mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
    x, y, bw, bh = det["bbox"]  # type: ignore
    cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
    return mask


def draw_mask_contours(
    img: np.ndarray,
    mask: np.ndarray,
    color: t.Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Return a copy of ``img`` with ``mask`` contours overlaid."""

    if cv2 is None:  # pragma: no cover - handled gracefully
        raise ImportError("OpenCV is required for drawing mask contours")

    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def mask_text_regions(
    img: np.ndarray,
    rects: t.Sequence[t.Tuple[int, int, int, int]],
    *,
    expand: int = 5,
    left_clamp: int = 25,
    inpaint: bool = False,
    inpaint_model=None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Return ``img`` with text rectangles removed and the binary mask."""

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        left = max(0, x0 - expand)
        top = max(0, y0 - expand)
        right = min(w, x1 + expand)
        bottom = min(h, y1 + expand)
        if x0 <= left_clamp:
            left = 0
        mask[top:bottom, left:right] = 255

    if inpaint and mask.any():
        pil_img = Image.fromarray(img[:, :, ::-1])
        pil_mask = Image.fromarray(mask)
        if inpaint_model is None:
            try:
                from simple_lama_inpainting import SimpleLama  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
                raise ModuleNotFoundError(
                    "simple-lama-inpainting is unavailable. Install with `pip install simple-lama-inpainting`"
                ) from exc
            inpaint_model = SimpleLama()
        inpainted = inpaint_model(pil_img, pil_mask)
        result = np.array(inpainted)[:, :, ::-1].copy()
    else:
        result = img.copy()
        result[mask > 0] = 0

    return result, mask


def create_text_mask(
    img_path,
    mask_path,
    ocr=None,
    preprocess=True,
    expand=5,
    left_clamp=25,
    inpaint_path=None,
    inpaint_model=None,
    debug_path=None,
):
    """Create a text mask for ``img_path`` and save to ``mask_path``."""

    if ocr is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError(
                "PaddleOCR is unavailable. Install paddlepaddle with `pip install \"paddlepaddle<3\"`"
            ) from exc
        ocr = PaddleOCR(
            det=True,
            rec=False,
            use_angle_cls=False,
            show_log=False,
            lang="en",
            det_db_thresh=0.25,
            det_db_box_thresh=0.38,
            det_db_unclip_ratio=2.2,
        )

    if cv2 is not None:
        bgr = cv2.imread(os.fspath(img_path))
    else:
        with Image.open(img_path) as _img:
            bgr = np.array(_img.convert("RGB"))[:, :, ::-1]

    scales = (1.0, 1.5, 2.0) if preprocess else (1.0,)
    rects, _ = detect_text_boxes(
        bgr,
        ocr,
        scales=scales,
        restrict_to_edges=True,
        suppress_glare=preprocess,
        preprocess=preprocess,
    )

    ui_mask = detect_ui_squares(bgr)

    inpaint = inpaint_path is not None
    processed, text_arr = mask_text_regions(
        bgr,
        rects,
        expand=expand,
        left_clamp=left_clamp,
        inpaint=inpaint,
        inpaint_model=inpaint_model,
    )

    combined = np.maximum(text_arr, ui_mask) if ui_mask is not None else text_arr
    _save_image(Image.fromarray(combined), mask_path)

    if debug_path is not None:
        dbg = bgr.copy()
        if ui_mask is not None:
            dbg = draw_mask_contours(dbg, ui_mask, color=(0, 255, 0))
        dbg = draw_mask_contours(dbg, text_arr, color=(0, 0, 255))
        _save_image(Image.fromarray(dbg[:, :, ::-1]), debug_path)

    if inpaint:
        _save_image(Image.fromarray(processed[:, :, ::-1]), inpaint_path)


def create_text_masks(
    src_dir,
    dest_dir,
    ocr=None,
    preprocess=True,
    expand=5,
    left_clamp=25,
    inpaint_dir=None,
    inpaint_model=None,
    debug_dir=None,
):
    """Generate text masks for all images in ``src_dir``."""

    src_dir = os.fspath(src_dir)
    dest_dir = os.fspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    if inpaint_dir is not None:
        inpaint_dir = os.fspath(inpaint_dir)
        os.makedirs(inpaint_dir, exist_ok=True)
    if debug_dir is not None:
        debug_dir = os.fspath(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)

    if ocr is None:
        from paddleocr import PaddleOCR  # type: ignore

        ocr = PaddleOCR(
            det=True,
            rec=False,
            use_angle_cls=False,
            show_log=False,
            lang="en",
            det_db_thresh=0.25,
            det_db_box_thresh=0.38,
            det_db_unclip_ratio=2.2,
        )

    for name in os.listdir(src_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            continue
        src_path = os.path.join(src_dir, name)
        base = os.path.splitext(name)[0]
        dest_path = os.path.join(dest_dir, f"{base}_mask.png")
        inpaint_path = (
            os.path.join(inpaint_dir, name) if inpaint_dir is not None else None
        )
        debug_path = (
            os.path.join(debug_dir, name) if debug_dir is not None else None
        )

        if cv2 is not None:
            bgr = cv2.imread(src_path)
        else:
            with Image.open(src_path) as _img:
                bgr = np.array(_img.convert("RGB"))[:, :, ::-1]

        scales = (1.0, 1.5, 2.0) if preprocess else (1.0,)
        rects, _ = detect_text_boxes(
            bgr,
            ocr,
            scales=scales,
            restrict_to_edges=True,
            suppress_glare=preprocess,
            preprocess=preprocess,
        )
        ui_mask = detect_ui_squares(bgr)
        inpaint = inpaint_path is not None
        processed, text_arr = mask_text_regions(
            bgr,
            rects,
            expand=expand,
            left_clamp=left_clamp,
            inpaint=inpaint,
            inpaint_model=inpaint_model,
        )
        combined = np.maximum(text_arr, ui_mask) if ui_mask is not None else text_arr
        _save_image(Image.fromarray(combined), dest_path)

        if debug_path is not None:
            dbg = bgr.copy()
            if ui_mask is not None:
                dbg = draw_mask_contours(dbg, ui_mask, color=(0, 255, 0))
            dbg = draw_mask_contours(dbg, text_arr, color=(0, 0, 255))
            _save_image(Image.fromarray(dbg[:, :, ::-1]), debug_path)

        if inpaint:
            _save_image(Image.fromarray(processed[:, :, ::-1]), inpaint_path)


def estimate_background_color(image: np.ndarray, region_frac: float = 0.25) -> Tuple[int, int, int]:
    """Estimate median colour of the bottom-left region of ``image``."""

    h, w = image.shape[:2]
    region = image[int(h * (1 - region_frac)) :, : int(w * region_frac)]
    pixels = region.reshape(-1, 3)
    median_color = np.median(pixels, axis=0).astype(np.uint8)
    return tuple(int(c) for c in median_color)


def is_color_similar(color: t.Sequence[int], ref=(135, 162, 18), tol: int = 35) -> bool:
    """Return ``True`` if ``color`` is within ``tol`` of ``ref`` for all channels."""

    return all(abs(int(c) - int(r)) <= tol for c, r in zip(color, ref))


def mask_full_viewport_by_expanded_box(
    image: np.ndarray,
    color_ref: t.Sequence[int],
    color_tol: int = 30,
    min_area: int = 1000,
    margin: int = 5,
) -> np.ndarray:
    """Create mask covering a bottom-left viewport matching ``color_ref``."""

    if cv2 is None:  # pragma: no cover - handled gracefully
        raise ImportError("OpenCV is required for viewport masking")

    lower = np.array([max(0, c - color_tol) for c in color_ref], dtype=np.uint8)
    upper = np.array([min(255, c + color_tol) for c in color_ref], dtype=np.uint8)
    color_mask = cv2.inRange(image, lower, upper)

    h, w = image.shape[:2]
    quadrant_mask = np.zeros((h, w), dtype=np.uint8)
    quadrant_mask[int(h / 2) :, : int(w / 2)] = 255
    combined = cv2.bitwise_and(color_mask, quadrant_mask)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    final_mask = np.zeros_like(combined)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x = max(0, x - margin)
            y = max(0, y - margin)
            bw = min(w - x, bw + 2 * margin)
            bh = min(h - y, bh + 2 * margin)
            cv2.rectangle(
                final_mask, (x, y), (x + bw, y + bh), 255, thickness=cv2.FILLED
            )

    return final_mask


def mask_viewport_auto_color_safe(
    image: np.ndarray,
    color_tol: int = 30,
    min_area: int = 1000,
    margin: int = 5,
    ref_color: t.Sequence[int] = (135, 162, 18),
    ref_tol: int = 35,
) -> Tuple[np.ndarray, Tuple[int, int, int], bool]:
    """Mask bottom-left viewport when its background colour looks like ``ref_color``."""

    color_ref = estimate_background_color(image)
    if not is_color_similar(color_ref, ref_color, tol=ref_tol):
        return image.copy(), color_ref, False

    mask = mask_full_viewport_by_expanded_box(
        image, color_ref, color_tol, min_area, margin
    )
    masked_img = image.copy()
    masked_img[mask == 255] = 0
    return masked_img, tuple(int(c) for c in color_ref), True


__all__ = [
    "mask_corner_artifacts",
    "crop_and_mask_corner_artifacts",
    "merge_rects",
    "preprocess_for_text",
    "detect_text_boxes",
    "detect_ui_square_row",
    "detect_ui_squares",
    "draw_mask_contours",
    "mask_text_regions",
    "create_text_mask",
    "create_text_masks",
    "estimate_background_color",
    "is_color_similar",
    "mask_full_viewport_by_expanded_box",
    "mask_viewport_auto_color_safe",
]

