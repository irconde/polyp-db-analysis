"""Utilities for preprocessing the PICCOLO dataset.

This module implements the workflow described in
``piccolo-database-analysis.ipynb``.  In addition to normalising column names
it crops padded polyp images, derives bounding boxes from the corresponding
segmentation masks and records the image dimensions of the cropped samples.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
import cv2
import logging

from .base import DatasetConverter
from ..utils import (
    crop_image_padding,
    get_image_size,
    move_images_parallel,
    validate_conversion,
)


logger = logging.getLogger(__name__)


def _collect_metadata(src_dir: Path | str, metadata_filename: str = "clinical metadata_release0.1.csv") -> pd.DataFrame:
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

    src_dir = Path(src_dir)
    df = pd.read_csv(src_dir / metadata_filename,
                     delimiter=";", encoding="latin1")
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


def _expand_frames(src_dir: Path | str, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Gather frame level information from the dataset structure.

    Args:
        src_dir: Root directory of the PICCOLO dataset.
        meta_df: DataFrame with per-lesion metadata.

    Returns:
        DataFrame with one row per image frame including dataset split and
        key polyp descriptors.
    """

    frames = []
    src_dir = Path(src_dir)
    for split in ["test", "train", "validation"]:
        polyp_dir = src_dir / split / "polyps"
        if not polyp_dir.is_dir():
            continue
        files = [f.name for f in polyp_dir.iterdir() if f.name.endswith(".png")]
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
    meta_unique = meta_df.drop_duplicates(
        subset=["CODE - LESION", "CODE - VIDEO"])
    df = frame_df.merge(
        meta_unique, on=["CODE - LESION", "CODE - VIDEO"], how="left")
    df = df.dropna(subset=["PARIS CLASSIFICATION"])
    df = df.rename(
        columns={"PARIS CLASSIFICATION": "SHAPE", "POLYP SIZE (mm)": "SIZE"})
    df["SHAPE"] = (
        df["SHAPE"]
        .str.replace("l", "I", regex=False)
        .str.replace("Ips", "Isp", regex=False)
        .str.extract(r"(Ip|Isp|Is|IIa|IIb|IIc|III)")
    )
    return df


def _derive_bboxes(df: pd.DataFrame, src_dir: Path | str, work_dir: Path | str) -> pd.DataFrame:
    """Crop images, compute bounding boxes and image dimensions.

    Args:
        df: DataFrame produced by :func:`_expand_frames`.
        src_dir: Root directory of the PICCOLO dataset.
        work_dir: Temporary directory for intermediate cropped files.

    Returns:
        DataFrame containing bounding boxes and image dimensions for each
        cropped frame. Images without masks are filtered out.
    """

    src_dir = Path(src_dir)
    work_dir = Path(work_dir)
    cropped_frame_dir = work_dir / "cropped_polyps"
    cropped_mask_dir = work_dir / "cropped_masks"
    cropped_frame_dir.mkdir(parents=True, exist_ok=True)
    cropped_mask_dir.mkdir(parents=True, exist_ok=True)

    def _process(row) -> tuple[str, int, int]:
        frame_path = src_dir / row.SET / "polyps" / row.FRAME
        mask_path = (
            src_dir / row.SET / "masks" / row.FRAME.replace(
                ".png", "_Corrected.tif")
        )
        cropped_frame_path = cropped_frame_dir / row.FRAME
        cropped_mask_path = cropped_mask_dir / \
            row.FRAME.replace(".png", ".tif")

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

    def extract_color_histogram(image):
        # Convert image to RGB (OpenCV loads images in BGR)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # Compute the color histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [
                            8, 8, 8], [0, 256, 0, 256, 0, 256])
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    for idx, frame in enumerate(df["FRAME"]):
        path = Path(img_dir) / frame
        try:
            with Image.open(path) as img:
                hist = extract_color_histogram(img)
                features.append(hist)
                indices.append(idx)
        except Exception as exc:
            logger.debug("Unable to read %s: %s", frame, exc)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(features)
    labels = kmeans.labels_

    # Add the classification results to a new column in the DataFrame
    image_types = pd.Series(pd.NA, index=df.index)
    for idx, label in zip(indices, labels):
        image_types.at[idx] = "WL" if label == 0 else "NBI"
    return image_types


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


class PiccoloConverter(DatasetConverter):
    """Converter for the PICCOLO dataset."""

    def __init__(
        self,
        src_dir: Path | str,
        dest_dir: Path | str,
        metadata_filename: str = "clinical metadata_release0.1.csv",
    ) -> None:
        super().__init__(src_dir, dest_dir, metadata_filename=metadata_filename)
        self.metadata_filename = metadata_filename
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._img_dir: Path | None = None

    def load_metadata(self) -> pd.DataFrame:
        meta = _collect_metadata(self.src_dir, self.metadata_filename)
        frames = _expand_frames(self.src_dir, meta)
        logger.info("Found %d frames with polyp annotations", len(frames))
        return frames

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def compute_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        self._tmpdir = tempfile.TemporaryDirectory()
        work_dir = Path(self._tmpdir.name)
        df = _derive_bboxes(df, self.src_dir, work_dir)
        self._img_dir = work_dir / "cropped_polyps"
        df["IMAGE TYPE"] = _infer_image_type(self._img_dir, df)
        df = _finalise_metadata(df)
        return df

    def write_outputs(self, df: pd.DataFrame) -> None:
        out_dir = self.dest_dir / "piccolo"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_path = out_dir / "annotations.xlsx"
        df.to_excel(frames_path, index=False)
        if self._img_dir is None:
            raise RuntimeError("compute_dimensions must be called before write_outputs")
        move_images_parallel(self._img_dir, out_dir / "img", df)
        validate_conversion(out_dir, df)
        logger.info("Finished PICCOLO conversion with %d entries", len(df))
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


def convert_to_standard_layout(
    src_dir: Path | str, dest_dir: Path | str, metadata_filename: str = "clinical metadata_release0.1.csv"
) -> None:
    """Backward compatible wrapper around :class:`PiccoloConverter`."""

    converter = PiccoloConverter(src_dir, dest_dir, metadata_filename=metadata_filename)
    converter.convert()
