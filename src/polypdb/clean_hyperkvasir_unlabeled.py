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

from .utils import detect_text_boxes, mask_viewport_auto_color_safe


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


def _mask_text_regions(
    img: np.ndarray,
    ocr,
    expand: int = 5,
    left_clamp: int = 25,
) -> np.ndarray:
    """Detect text boxes in ``img`` and mask them with black rectangles."""

    rects, _ = detect_text_boxes(
        img,
        ocr,
        scales=(1.0, 1.5, 2.0),
        restrict_to_edges=True,
        suppress_glare=True,
        preprocess=True,
    )
    h, w = img.shape[:2]
    for x0, y0, x1, y1 in rects:
        left = max(0, x0 - expand)
        top = max(0, y0 - expand)
        right = min(w, x1 + expand)
        bottom = min(h, y1 + expand)
        if x0 <= left_clamp:
            left = 0
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), thickness=cv2.FILLED)
    return img


def _clean_frame(img: np.ndarray, ocr) -> np.ndarray:
    """Apply masking and cropping operations to ``img``."""

    img, _, _ = mask_viewport_auto_color_safe(img)
    img = _mask_text_regions(img, ocr)
    if img.shape[0] > 10 and img.shape[1] > 10:
        img = img[5:-5, 5:-5]
    img = _crop_padding(img)
    return img


def batch_clean_hyperkvasir_unlabeled(input_dir: str, output_dir: str) -> None:
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
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    for img_path in sorted(src.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: couldn't read {img_path}")
            continue
        cleaned = _clean_frame(image, ocr)
        cv2.imwrite(str(dst / img_path.name), cleaned)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Hyper-Kvasir-unlabeled frames by masking viewports and text, then cropping.",
    )
    parser.add_argument("input_dir", help="Directory with raw frames")
    parser.add_argument("output_dir", help="Directory to save cleaned frames")
    args = parser.parse_args()
    batch_clean_hyperkvasir_unlabeled(args.input_dir, args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
