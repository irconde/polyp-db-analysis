"""CLI to clean Hyper-Kvasir-unlabeled frames.

The pipeline performs the following operations on each image:

1. Mask a small bottom-left viewport when detected.
2. Mask textual overlays using PaddleOCR detections.
3. Crop a 5 pixel border around the image.
4. Remove padding pixels of a specified colour.
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple, Optional

import cv2  # type: ignore
import numpy as np

from .utils import (
    detect_text_boxes,
    detect_ui_squares,
    mask_text_regions,
    mask_viewport_auto_color_safe,
)
from .br_padding_fill import fill_br_padding


def _estimate_padding_rgb(img: np.ndarray, band_frac: float = 0.04) -> Tuple[int, int, int]:
    """Robust per-image padding colour from a thin border band (median, RGB)."""
    h, w = img.shape[:2]
    bw, bh = max(1, int(w * band_frac)), max(1, int(h * band_frac))
    border = np.concatenate(
        [
            img[:bh, :, :].reshape(-1, 3),
            img[-bh:, :, :].reshape(-1, 3),
            img[:, :bw, :].reshape(-1, 3),
            img[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return tuple(np.median(border, axis=0).astype(np.uint8).tolist())  # type: ignore[return-value]


def _crop_padding(
    img: np.ndarray,
    padding_color: Optional[Sequence[int]] = (15, 15, 15),
    threshold: int = 20,
    min_row_frac: float = 0.01,
    min_col_frac: float = 0.01,
    erode_iter: int = 2,
    open_kernel: int = 3,
) -> np.ndarray:
    """
    Crop uniform padding while ignoring small stains/noise.

    - padding_color=None → auto-estimate from border.
    - threshold: per-channel tolerance around the padding colour (RGB box).
    - min_row_frac/min_col_frac: minimum fraction of "content" pixels required
      for a row/column to count as non-padding.
    - erode_iter/open_kernel: morphology to remove isolated specks.
    """
    if padding_color is None:
        padding_color = _estimate_padding_rgb(img)

    color = np.array(padding_color, dtype=np.int16)
    lower = np.clip(color - threshold, 0, 255)
    upper = np.clip(color + threshold, 0, 255)

    # padding-like pixels
    pad_like = np.all((img >= lower) & (img <= upper), axis=-1)

    # content mask (uint8) and clean small specks
    content = (~pad_like).astype(np.uint8) * 255
    if erode_iter > 0:
        content = cv2.erode(content, np.ones((3, 3), np.uint8), iterations=erode_iter)
    if open_kernel > 0:
        content = cv2.morphologyEx(content, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))

    h, w = content.shape
    min_nonpad_row = max(1, int(w * min_row_frac))
    min_nonpad_col = max(1, int(h * min_col_frac))

    row_ok = (content > 0).sum(axis=1) >= min_nonpad_row
    col_ok = (content > 0).sum(axis=0) >= min_nonpad_col

    # relaxed fallback if morphology removed too much
    if not row_ok.any() or not col_ok.any():
        row_ok = (~pad_like).sum(axis=1) >= min_nonpad_row
        col_ok = (~pad_like).sum(axis=0) >= min_nonpad_col
        if not row_ok.any() or not col_ok.any():
            return img  # nothing safe to crop

    top = int(np.argmax(row_ok))
    bottom = int(len(row_ok) - np.argmax(row_ok[::-1]))
    left = int(np.argmax(col_ok))
    right = int(len(col_ok) - np.argmax(col_ok[::-1]))
    return img[top:bottom, left:right]


def _clean_frame(
    img: np.ndarray,
    ocr,
    *,
    inpaint_text: bool = False,
    inpaint_model=None,
    draw_ui_bbox: bool = False,
) -> np.ndarray:
    """Apply masking and cropping operations to ``img``.

    Parameters
    ----------
    img:
        Input BGR image.
    ocr:
        PaddleOCR instance or ``None``.
    inpaint_text:
        When ``True`` detected text regions are inpainted instead of filled
        with black.
    inpaint_model:
        Optional inpainting model used when ``inpaint_text`` is ``True``.
    draw_ui_bbox:
        When ``True`` draw the bounding box of detected UI squares onto the
        image for debugging.
    """

    img, _, _ = mask_viewport_auto_color_safe(img)
    rects, _ = detect_text_boxes(
        img,
        ocr,
        scales=(0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),
        restrict_to_edges=False,
        suppress_glare=False,
        preprocess=False,
    )
    ui_rect = None
    ui_mask = detect_ui_squares(img)
    if ui_mask is not None and ui_mask.any():
        ys, xs = np.where(ui_mask > 0)
        ui_rect = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        rects = list(rects) + [ui_rect]
    img, _ = mask_text_regions(
        img,
        rects,
        expand=10,
        left_clamp=25,
        inpaint=inpaint_text,
        inpaint_model=inpaint_model,
    )
    if draw_ui_bbox and ui_rect is not None:
        # ``img`` may be a non-contiguous view when inpainting is enabled, which
        # OpenCV drawing functions do not accept. Ensure contiguity before
        # drawing the bounding box for debugging.
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (ui_rect[0], ui_rect[1]), (ui_rect[2], ui_rect[3]), (0, 255, 0), 1)
    if img.shape[0] > 10 and img.shape[1] > 10:
        img = img[5:-5, 5:-5]
    img = _crop_padding(img)
    img = fill_br_padding(img)
    return img


def batch_clean_hyperkvasir_unlabeled(
    input_dir: str,
    output_dir: str,
    *,
    inpaint_text: bool = False,
    resume: bool = True,
) -> None:
    """Clean all images in ``input_dir`` and write them to ``output_dir``.

    When ``resume`` is ``True`` existing files in ``output_dir`` are skipped,
    allowing interrupted runs to continue without restarting from scratch.
    """

    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "PaddleOCR is required for text detection. Install paddlepaddle with `pip install \"paddlepaddle<3\"`"
        ) from exc

    ocr = PaddleOCR(use_angle_cls=False, lang="en")
    inpaint_model = None
    if inpaint_text:
        try:
            from simple_lama_inpainting import SimpleLama  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "simple-lama-inpainting is required for text inpainting. Install with `pip install simple-lama-inpainting`",
            ) from exc
        inpaint_model = SimpleLama()
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_paths = [p for p in sorted(src.iterdir()) if p.suffix.lower() in extensions]
    if resume:
        img_paths = [p for p in img_paths if not (dst / p.name).exists()]

    for img_path in sorted(src.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: couldn't read {img_path}")
            continue
        cleaned = _clean_frame(
            image,
            ocr,
            inpaint_text=inpaint_text,
            inpaint_model=inpaint_model,
        )
        cv2.imwrite(str(dst / img_path.name), cleaned)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Hyper-Kvasir-unlabeled frames by masking viewports and text, then cropping.",
    )
    parser.add_argument("input_dir", help="Directory with raw frames")
    parser.add_argument("output_dir", help="Directory to save cleaned frames")
    parser.add_argument(
        "--inpaint",
        action="store_true",
        help="Fill masked text regions using SimpleLama inpainting.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess all images even if outputs already exist.",
    )
    args = parser.parse_args()
    batch_clean_hyperkvasir_unlabeled(
        args.input_dir,
        args.output_dir,
        inpaint_text=args.inpaint,
        resume=not args.no_resume,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
