"""Preprocessing helpers for the SUN dataset.

The notebook ``sun-database-analysis.ipynb`` performs a number of additional
steps beyond simple column renaming.  Here we extract image sizes, ensure all
bounding boxes lie within image bounds and preserve the white-light modality
information when exporting the dataset.
"""

import os
import pandas as pd

from ..utils import get_image_size, move_images


def _clip_bbox(bbox: str, width: int, height: int) -> str:
    boxes = []
    for bb in bbox.split(" "):
        if not bb:
            continue
        x1, y1, x2, y2 = map(int, bb.split(","))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        boxes.append(f"{x1},{y1},{x2},{y2}")
    return " ".join(boxes)


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert SUN dataset to the standard directory layout."""

    df = pd.read_excel(os.path.join(src_dir, "annotations.xlsx"))
    df["FRAME"] = df["FRAME"].astype(str)
    df["FRAME"] = df["FRAME"].apply(
        lambda x: x if x.lower().endswith(".png") else f"{x}.png"
    )

    widths = []
    heights = []
    for frame in df["FRAME"]:
        w, h = get_image_size(os.path.join(src_dir, "img_all", frame))
        widths.append(w)
        heights.append(h)
    df["WIDTH"] = widths
    df["HEIGHT"] = heights

    if "BOUNDING BOX" in df.columns:
        df["BOUNDING BOX"] = df.apply(
            lambda r: _clip_bbox(r["BOUNDING BOX"], r["WIDTH"], r["HEIGHT"]), axis=1
        )

    df["IMAGE TYPE"] = "WL"

    out_dir = os.path.join(dest_dir, "sun")
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    move_images(os.path.join(src_dir, "img_all"), os.path.join(out_dir, "img"), df)
