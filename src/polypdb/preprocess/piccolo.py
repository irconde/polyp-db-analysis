"""Utilities for preprocessing the PICCOLO dataset.

This module implements the workflow described in
``piccolo-database-analysis.ipynb``.  In addition to normalising column names
it crops padded polyp images, derives bounding boxes from the corresponding
segmentation masks and records the image dimensions of the cropped samples.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
import logging

from ..utils import (
    crop_image_padding,
    get_image_size,
    move_images_parallel,
    validate_conversion,
)


logger = logging.getLogger(__name__)
def _collect_metadata(src_dir: str, metadata_filename: str = "clinical metadata_release0.1.csv") -> pd.DataFrame:
    """Load and clean the raw annotation CSV.

    The original CSV uses ``;`` as a delimiter and contains a number of empty
    columns and placeholder values. The function returns a cleaned DataFrame
    with unnecessary columns removed and key identifiers formatted.

    Args:
        src_dir: Root directory of the PICCOLO dataset.
        metadata_filename: Name of the CSV file with polyp metadata.

    Returns:
        ``pandas.DataFrame`` with normalised metadata columns.
    """

    df = pd.read_csv(os.path.join(src_dir, metadata_filename), delimiter=";", encoding="latin1")
    df = df.replace("-", pd.NA).dropna(how="all").dropna(axis=1, how="all")
    df = df.drop(
        columns=[
            "HISTOLOGICAL STRATIFICATION",
            "LITERAL DIAGNOSIS (Pathologist)",
            "PRELIMINAR DIAGNOSIS\xa0 (Gastroenterologist)",
            "NICE ",
        ],
        errors="ignore",
    )
    df["CODE - LESION"] = df["CODE - LESION"].fillna(0).apply(lambda x: f"{int(x):03d}")
    if "CODE - VIDEO" in df.columns:
        df["CODE - VIDEO"] = df["CODE - VIDEO"].astype(str).str.zfill(2)
    df["NUMBER OF POLYPS OF INTEREST (N)"] = df[
        "NUMBER OF POLYPS OF INTEREST (N)"
    ].fillna(0).astype(int)
    return df


def _expand_frames(src_dir: str, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Gather frame level information from the dataset structure.

    Args:
        src_dir: Root directory of the PICCOLO dataset.
        meta_df: DataFrame with per-lesion metadata.

    Returns:
        DataFrame with one row per image frame including dataset split and
        key polyp descriptors.
    """

    rows: List[dict] = []
    for split in ["test", "train", "validation"]:
        polyp_dir = os.path.join(src_dir, split, "polyps")
        if not os.path.isdir(polyp_dir):
            continue
        for fname in os.listdir(polyp_dir):
            if not fname.endswith(".png"):
                continue
            lesion, video, _ = fname.split("_")
            match = meta_df[
                (meta_df["CODE - LESION"] == lesion)
                & (meta_df["CODE - VIDEO"] == video)
            ]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row.update({"FRAME": fname, "SET": split})
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["PARIS CLASSIFICATION"])
    df = df.rename(columns={"PARIS CLASSIFICATION": "SHAPE", "POLYP SIZE (mm)": "SIZE"})
    df["SHAPE"] = (
        df["SHAPE"]
        .str.replace("l", "I", regex=False)
        .str.replace("Ips", "Isp", regex=False)
        .str.extract(r"(Ip|Isp|Is|IIa|IIb|IIc|III)")
    )
    return df


def _derive_bboxes(df: pd.DataFrame, src_dir: str) -> pd.DataFrame:
    """Crop images, compute bounding boxes and image dimensions.

    Args:
        df: DataFrame produced by :func:`_expand_frames`.
        src_dir: Root directory of the PICCOLO dataset.

    Returns:
        DataFrame containing bounding boxes and image dimensions for each
        cropped frame. Images without masks are filtered out.
    """

    cropped_frame_dir = os.path.join(src_dir, "cropped_polyps")
    cropped_mask_dir = os.path.join(src_dir, "cropped_masks")
    os.makedirs(cropped_frame_dir, exist_ok=True)
    os.makedirs(cropped_mask_dir, exist_ok=True)

    def _process(row) -> tuple[str, int, int]:
        frame_path = os.path.join(src_dir, row.SET, "polyps", row.FRAME)
        mask_path = os.path.join(
            src_dir, row.SET, "masks", row.FRAME.replace(".png", "_Corrected.tif")
        )
        cropped_frame_path = os.path.join(cropped_frame_dir, row.FRAME)
        cropped_mask_path = os.path.join(cropped_mask_dir, row.FRAME.replace(".png", ".tif"))

        crop_image_padding(
            frame_path,
            cropped_frame_path,
            mask_path=mask_path,
            cropped_mask_path=cropped_mask_path,
            padding_color=(15, 15, 15),
            threshold=20,
        )

        w, h = get_image_size(cropped_frame_path)

        try:
            with Image.open(cropped_mask_path) as mask_img:
                mask = np.array(mask_img) > 0
        except FileNotFoundError:
            return "", w, h

        props = regionprops(label(mask))
        boxes = [
            f"{p.bbox[1]},{p.bbox[0]},{p.bbox[3]},{p.bbox[2]}" for p in props
        ]
        return " ".join(boxes), w, h

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_process, df.itertuples(index=False)))

    if results:
        bboxes, widths, heights = zip(*results)
    else:
        bboxes, widths, heights = [], [], []

    df = df.copy()
    df["BOUNDING BOX"] = list(bboxes)
    df["WIDTH"] = list(widths)
    df["HEIGHT"] = list(heights)
    df = df[df["BOUNDING BOX"] != ""]
    return df.drop(columns=["SET"])


def convert_to_standard_layout(
    src_dir: str, dest_dir: str, metadata_filename: str = "clinical metadata_release0.1.csv"
) -> None:
    """Convert PICCOLO dataset to the standard directory layout.

    Args:
        src_dir: Root directory of the PICCOLO dataset.
        dest_dir: Directory where the converted dataset will be written.
        metadata_filename: Name of the CSV file with polyp metadata.

    Returns:
        None
    """

    logger.info("Converting PICCOLO dataset from %s to %s", src_dir, dest_dir)
    meta = _collect_metadata(src_dir, metadata_filename)
    frames = _expand_frames(src_dir, meta)
    frames = _derive_bboxes(frames, src_dir)

    out_dir = os.path.join(dest_dir, "piccolo")
    os.makedirs(out_dir, exist_ok=True)
    frames.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    move_images_parallel(
        os.path.join(src_dir, "cropped_polyps"), os.path.join(out_dir, "img"), frames
    )
    validate_conversion(out_dir, frames)
    logger.info("Finished PICCOLO conversion with %d entries", len(frames))

    # Cleanup intermediate folders after processing
    import shutil
    intermediate_dirs = [
        os.path.join(src_dir, "cropped_polyps"),
        os.path.join(src_dir, "cropped_masks"),
    ]
    for d in intermediate_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)

