from __future__ import annotations

"""Utilities for preprocessing the PolypGen dataset.

This module processes the single-frame data provided as part of the
PolypGen2021 multi-centre dataset.  Each centre directory contains the
``images_CN`` folder with image frames and a matching ``bbox_CN`` directory
with one text file per image describing the bounding boxes in a simple
VOC-like format.  Empty annotation files indicate that no polyps are present in
that frame.

The :func:`convert_to_standard_layout` function converts these files into the
canonical layout used throughout the project, copying the images and producing
an ``annotations.xlsx`` file with normalised bounding box coordinates.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..utils import get_image_size, move_images_parallel, validate_conversion

logger = logging.getLogger(__name__)


def _parse_bbox_file(path: str) -> str:
    """Return bounding boxes from ``path`` normalised to ``x1,y1,x2,y2`` format.

    Lines in the file may optionally start with a class label.  Coordinates are
    interpreted either as ``x_min y_min x_max y_max`` or ``x_min y_min width height``.
    Multiple boxes are joined using spaces.  Missing files or empty annotations
    result in an empty string.
    """

    if not os.path.exists(path):
        return ""

    boxes: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
            # Drop non-numeric class label when present
            if parts and not re.match(r"^-?\d+(?:\.\d+)?$", parts[0]):
                parts = parts[1:]
            if len(parts) < 4:
                continue
            nums = list(map(float, parts[:4]))
            x1, y1, x2, y2 = nums
            if x2 <= x1 or y2 <= y1:
                # Interpret as x, y, width, height
                x1, y1, w, h = nums
                x2 = x1 + w
                y2 = y1 + h
            boxes.append(
                f"{int(round(x1))},{int(round(y1))},{int(round(x2))},{int(round(y2))}"
            )
    return " ".join(boxes)


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert PolypGen dataset to the standard directory layout.

    Only single-frame data under ``data_C1`` … ``data_C6`` is processed.  The
    resulting images are copied to ``dest_dir/polypgen/img`` and an
    ``annotations.xlsx`` file is created with one row per frame including the
    centre identifier, bounding boxes, image dimensions and metadata columns
    required by the rest of the pipeline.  All images are white-light and the
    dataset does not provide a polyp identifier.
    """

    logger.info("Converting PolypGen dataset from %s to %s", src_dir, dest_dir)

    rows: list[dict] = []
    for centre in range(1, 7):
        centre_name = f"C{centre}"
        centre_dir = os.path.join(src_dir, f"data_{centre_name}")
        img_dir = os.path.join(centre_dir, f"images_{centre_name}")
        bbox_dir = os.path.join(centre_dir, f"bbox_{centre_name}")
        if not os.path.isdir(img_dir) or not os.path.isdir(bbox_dir):
            continue
        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = os.path.splitext(img_file)[0]
            bbox_file = os.path.join(bbox_dir, base + ".txt")
            bbox = _parse_bbox_file(bbox_file)
            src_path = os.path.join(img_dir, img_file)
            rows.append(
                {
                    "FRAME": img_file,
                    "POLYP ID": pd.NA,
                    "CENTER": centre_name,
                    "BOUNDING BOX": bbox,
                    "SRC_PATH": src_path,
                    "IMAGE TYPE": "WL",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        with ThreadPoolExecutor() as executor:
            sizes = list(executor.map(get_image_size, df["SRC_PATH"]))
        df["WIDTH"], df["HEIGHT"] = zip(*sizes)
    else:
        df["WIDTH"] = []
        df["HEIGHT"] = []

    out_dir = os.path.join(dest_dir, "polypgen")
    img_dest_dir = os.path.join(out_dir, "img")
    move_images_parallel(
        None,
        img_dest_dir,
        df,
        filename_col="FRAME",
        source_path_col="SRC_PATH",
        copy=True,
    )

    df = df.drop(columns=["SRC_PATH"], errors="ignore")
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    validate_conversion(out_dir, df)
    logger.info("Finished PolypGen conversion with %d entries", len(df))
