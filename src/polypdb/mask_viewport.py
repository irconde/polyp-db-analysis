"""CLI utility to mask small bottom-left viewports in images."""

import argparse
from pathlib import Path

import cv2  # type: ignore

from .utils import mask_viewport_auto_color_safe


def batch_mask_viewport(input_dir: str, output_dir: str, ext: str = ".jpg") -> None:
    """Process all images in ``input_dir`` masking bottom-left viewports.

    Parameters
    ----------
    input_dir:
        Folder containing input images.
    output_dir:
        Destination directory for masked images. It is created if missing.
    ext:
        File extension to filter images by (e.g. ``.jpg``).
    """

    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for img_path in src.glob(f"*{ext}"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: couldn't read {img_path}")
            continue
        masked_img, color, applied = mask_viewport_auto_color_safe(image)
        if applied:
            print(f"Masked: {img_path.name} (color: {tuple(color)})")
        else:
            print(f"Skipped: {img_path.name} (color: {tuple(color)})")
        cv2.imwrite(str(dst / img_path.name), masked_img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely mask bottom-left viewports from endoscopic images."
    )
    parser.add_argument("input_dir", type=str, help="Path to folder with input images")
    parser.add_argument("output_dir", type=str, help="Path to save masked images")
    parser.add_argument(
        "--ext", type=str, default=".jpg", help="Image file extension (e.g. .jpg, .png)"
    )
    args = parser.parse_args()

    batch_mask_viewport(args.input_dir, args.output_dir, args.ext)


if __name__ == "__main__":
    main()

