"""Utilities for preprocessing the PICCOLO dataset.

This module implements the workflow described in
``piccolo-database-analysis.ipynb``.  In addition to normalising column names
it crops padded polyp images, derives bounding boxes from the corresponding
segmentation masks and records the image dimensions of the cropped samples.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
import cv2
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
    df["CODE - LESION"] = (
        df["CODE - LESION"]
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )
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

    frames = []
    for split in ["test", "train", "validation"]:
        polyp_dir = os.path.join(src_dir, split, "polyps")
        if not os.path.isdir(polyp_dir):
            continue
        files = [f for f in os.listdir(polyp_dir) if f.endswith(".png")]
        if not files:
            continue
        frame_df = pd.DataFrame({"FRAME": files, "SET": split})
        parts = frame_df["FRAME"].str.split("_", expand=True)
        frame_df["CODE - LESION"] = parts[0]
        frame_df["CODE - VIDEO"] = parts[1]
        frames.append(frame_df)
    if not frames:
        return pd.DataFrame(columns=list(meta_df.columns) + ["FRAME", "SET"])

    frame_df = pd.concat(frames, ignore_index=True)
    meta_unique = meta_df.drop_duplicates(subset=["CODE - LESION", "CODE - VIDEO"])
    df = frame_df.merge(meta_unique, on=["CODE - LESION", "CODE - VIDEO"], how="left")
    df = df.dropna(subset=["PARIS CLASSIFICATION"])
    df = df.rename(columns={"PARIS CLASSIFICATION": "SHAPE", "POLYP SIZE (mm)": "SIZE"})
    df["SHAPE"] = (
        df["SHAPE"]
        .str.replace("l", "I", regex=False)
        .str.replace("Ips", "Isp", regex=False)
        .str.extract(r"(Ip|Isp|Is|IIa|IIb|IIc|III)")
    )
    return df


def _derive_bboxes(df: pd.DataFrame, src_dir: str, work_dir: str) -> pd.DataFrame:
    """Crop images, compute bounding boxes and image dimensions.

    Args:
        df: DataFrame produced by :func:`_expand_frames`.
        src_dir: Root directory of the PICCOLO dataset.
        work_dir: Temporary directory for intermediate cropped files.

    Returns:
        DataFrame containing bounding boxes and image dimensions for each
        cropped frame. Images without masks are filtered out.
    """

    cropped_frame_dir = os.path.join(work_dir, "cropped_polyps")
    cropped_mask_dir = os.path.join(work_dir, "cropped_masks")
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
    df = df[df["BOUNDING BOX"] != ""].reset_index(drop=True)
    return df.drop(columns=["SET"])


def _infer_image_type(img_dir: str, df: pd.DataFrame) -> pd.Series:
    """Infer whether each frame is captured using WL or NBI.

    A simple unsupervised approach based on colour histogram clustering is
    employed. The implementation mirrors the logic originally used in the
    ``piccolo-database-analysis`` notebook. Images are loaded, converted from
    OpenCV's default BGR ordering to RGB and an RGB colour histogram is
    extracted. K-means with two clusters is then used to separate white light
    (WL) from narrow band imaging (NBI) frames. The cluster with label ``0`` is
    assumed to correspond to WL as observed empirically.

    Parameters
    ----------
    img_dir:
        Directory containing cropped polyp images.
    df:
        DataFrame with a ``FRAME`` column identifying the images.

    Returns
    -------
    pandas.Series
        Series aligned with ``df`` containing the inferred image type for
        each frame (``'WL'`` or ``'NBI'``). Missing or unreadable images are
        reported as ``pd.NA``.
    """

    features: list[np.ndarray] = []
    indices: list[int] = []
    unreadable: list[int] = []
    for idx, frame in enumerate(df["FRAME"]):
        path = os.path.join(img_dir, frame)
        img = cv2.imread(path)
        if img is None:
            try:
                with Image.open(path) as pil_img:
                    img = np.array(pil_img)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning("Unable to read %s: %s", frame, exc)
                unreadable.append(idx)
                continue
        try:
            # OpenCV loads images in BGR format; convert to RGB so that the
            # histogram reflects the expected colour channels.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist = cv2.calcHist(
                [img],
                [0, 1, 2],
                None,
                [8, 8, 8],
                [0, 256, 0, 256, 0, 256],
            )
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
            indices.append(idx)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.debug("Failed to process %s: %s", frame, exc)

    if unreadable:
        bad_frames = ", ".join(f"{i}:{df['FRAME'][i]}" for i in unreadable)
        raise ValueError(f"Failed to read images at indices: {bad_frames}")

    series = pd.Series(pd.NA, index=range(len(df)), dtype="object")
    if not features:
        return series

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(features)
    for i, label in zip(indices, kmeans.labels_):
        series.iat[i] = "WL" if label == 0 else "NBI"
    return series


def _finalise_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant columns, rename fields and ensure numeric types."""

    df = df.rename(columns={"CODE - LESION": "POLYP ID"})
    df = df.drop(
        columns=[
            "CODE - VIDEO",
            "NUMBER OF POLYPS OF INTEREST (N)",
            "CURRENT POLYP ID (X/N)",
        ],
        errors="ignore",
    )
    for col in ["SIZE", "WIDTH", "HEIGHT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


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

    with tempfile.TemporaryDirectory() as work_dir:
        frames = _derive_bboxes(frames, src_dir, work_dir)

        img_dir = os.path.join(work_dir, "cropped_polyps")
        frames["IMAGE TYPE"] = _infer_image_type(img_dir, frames)
        frames = _finalise_metadata(frames)

        out_dir = os.path.join(dest_dir, "piccolo")
        os.makedirs(out_dir, exist_ok=True)
        frames.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
        move_images_parallel(img_dir, os.path.join(out_dir, "img"), frames)
        validate_conversion(out_dir, frames)
        logger.info("Finished PICCOLO conversion with %d entries", len(frames))

    # TemporaryDirectory context automatically removes intermediate files
