"""CLI to clean Hyper-Kvasir-unlabeled frames.

The pipeline performs the following operations on each image:

1. Mask a small bottom-left viewport when detected.
2. Mask textual overlays using PaddleOCR detections.
3. Crop a 5 pixel border around the image.
4. Remove padding pixels of a specified colour.
"""

import argparse
from pathlib import Path
import typing as t

import cv2  # type: ignore
import numpy as np

from .utils import (
    detect_text_boxes,
    detect_ui_squares,
    mask_text_regions,
    mask_viewport_auto_color_safe,
)


def _crop_padding(
    img: np.ndarray,
    padding_color: t.Sequence[int] = (15, 15, 15),
    threshold: int = 20,
) -> np.ndarray:
    """Crop uniform padding from ``img``.

    ``padding_color`` defines the RGB colour regarded as padding and ``threshold``
    specifies how much deviation per channel is tolerated.
    """

    color = np.array(padding_color, dtype=np.int16)
    lower = np.clip(color - threshold, 0, 255)
    upper = np.clip(color + threshold, 0, 255)

    close = np.all((img >= lower) & (img <= upper), axis=-1)
    mask = ~close
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    top = rows.argmax()
    bottom = len(rows) - rows[::-1].argmax()
    left = cols.argmax()
    right = len(cols) - cols[::-1].argmax()
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
        cv2.rectangle(img, (ui_rect[0], ui_rect[1]), (ui_rect[2], ui_rect[3]), (0, 255, 0), 1)
    if img.shape[0] > 10 and img.shape[1] > 10:
        img = img[5:-5, 5:-5]
    img = _crop_padding(img)
    return img


def batch_clean_hyperkvasir_unlabeled(
    input_dir: str,
    output_dir: str,
    *,
    inpaint_text: bool = False,
) -> None:
    """Clean all images in ``input_dir`` and write them to ``output_dir``."""

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
    args = parser.parse_args()
    batch_clean_hyperkvasir_unlabeled(
        args.input_dir,
        args.output_dir,
        inpaint_text=args.inpaint,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
