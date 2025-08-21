"""Preprocessing helpers for the SUN dataset.

The original SUN distribution provides a spreadsheet describing each polyp
and a set of text files with the bounding-box annotations.  The exploratory
``sun-database-analysis.ipynb`` notebook performs a fair amount of munging on
this data: normalising column names, fixing shape labels, parsing the text
annotations, clipping bounding boxes that extend outside the image and
recording the size of every frame.  The goal of this module is to replicate
that logic so that the dataset can be converted automatically without having
to rely on the notebook.
"""

import os
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from ..utils import get_image_size, move_images_parallel, validate_conversion


logger = logging.getLogger(__name__)


LOSSLESS_EXT = {".png", ".tif", ".tiff"}


def _pixel_hash(path: str) -> str:
    """Return a hash of the RGB pixel data found at ``path``."""
    with Image.open(path).convert("RGB") as im:
        arr = np.asarray(im)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _rank_key(path: str, width: int, height: int, filesize: int) -> tuple:
    """Ranking key used to pick the canonical frame within duplicates."""
    ext = os.path.splitext(path)[1].lower()
    lossless_rank = 0 if ext in LOSSLESS_EXT else 1
    return (-height * width, lossless_rank, -filesize, path)


def _drop_duplicate_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` without duplicate frames based on pixel content."""

    if df.empty:
        return df

    df = df.copy()
    df["_pixel_hash"] = df["SRC_PATH"].apply(_pixel_hash)
    df["_filesize"] = df["SRC_PATH"].apply(lambda p: os.path.getsize(p))

    duplicates: list[str] = []
    for _, group in df.groupby("_pixel_hash"):
        if len(group) <= 1:
            continue
        ranked = sorted(
            group.itertuples(),
            key=lambda r: _rank_key(r.SRC_PATH, r.WIDTH, r.HEIGHT, r._filesize),
        )
        duplicates.extend(r.SRC_PATH for r in ranked[1:])

    if duplicates:
        logger.info("Omitting %d duplicate frames", len(duplicates))
        df = df[~df["SRC_PATH"].isin(duplicates)]

    return df.drop(columns=["_pixel_hash", "_filesize"])


def _clip_bbox(bbox: str, width: int, height: int) -> str:
    """Clip ``bbox`` coordinates so that they lie within the image bounds.

    Args:
        bbox: One or more bounding boxes separated by spaces in
            ``x1,y1,x2,y2`` format.
        width: Width of the image used for clipping.
        height: Height of the image used for clipping.

    Returns:
        A space-separated string of bounding boxes with coordinates clamped
        to the image size.
    """

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


def _clip_bboxes(bboxes: pd.Series, widths: pd.Series, heights: pd.Series) -> pd.Series:
    """Vectorised clipping of bounding boxes to image dimensions.

    Args:
        bboxes: Series of space-separated ``x1,y1,x2,y2`` bounding boxes.
        widths: Series with corresponding image widths.
        heights: Series with corresponding image heights.

    Returns:
        Series with bounding boxes clamped to the image size.
    """

    result = []
    for bbox, w, h in zip(bboxes, widths, heights):
        if not bbox:
            result.append("")
            continue
        coords = np.fromstring(bbox.replace(" ", ","), sep=",", dtype=int).reshape(-1, 4)
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, w)
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, h)
        result.append(" ".join(",".join(map(str, c)) for c in coords))
    return pd.Series(result, index=bboxes.index)


def _extract_bboxes(line: str) -> tuple[str, str]:
    """Return the filename and bounding boxes encoded in ``line``.

    The annotation files contain one line per image where the first token is
    the image filename followed by ``x1,y1,x2,y2,0`` entries. The trailing
    ``0`` represents the class id (polyp) and is discarded.

    Args:
        line: Raw line from an annotation text file.

    Returns:
        A tuple containing the image filename and a space-separated string of
        bounding boxes.
    """

    parts = line.strip().split(" ")
    filename = parts[0]
    bbox_strings = []
    for bbox_str in parts[1:]:
        bbox_values = bbox_str.split(",")[:-1]  # drop class id
        bbox_strings.append(",".join(bbox_values))
    return filename, " ".join(bbox_strings)


def convert_to_standard_layout(
    src_dir: str, dest_dir: str, metadata_filename: str = "metadata.csv"
) -> None:
    """Convert SUN dataset to the standard directory layout.

    ``src_dir`` should contain the ``metadata_filename`` file as well as the
    ``sundatabase_positive`` folder with ``annotation_txt`` and ``case*``
    subdirectories. The resulting annotations and images are written under
    ``dest_dir/sun``.

    Args:
        src_dir: Root directory of the SUN dataset.
        dest_dir: Directory where the converted dataset will be written.
        metadata_filename: Name of the CSV file with polyp metadata.

    Returns:
        None
    """

    logger.info("Converting SUN dataset from %s to %s", src_dir, dest_dir)
    metadata_path = os.path.join(src_dir, metadata_filename)
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.columns = [c.upper() for c in metadata_df.columns]
    metadata_df["SHAPE"] = metadata_df["SHAPE"].replace("IIa(LST-NG)", "IIa")
    metadata_df = metadata_df.drop(columns=["PATHOLOGICAL DIAGNOSIS"], errors="ignore")

    annotation_txt_dir = os.path.join(
        src_dir, "sundatabase_positive", "annotation_txt"
    )

    def _parse_case_annotation(args: tuple[str, pd.Series]) -> list[dict]:
        """Parse a single case annotation file into per-frame rows."""

        txt_file, meta = args
        case_rows: list[dict] = []
        with open(txt_file, "r") as f:
            for line in f:
                filename, bbox = _extract_bboxes(line)
                case_rows.append(
                    {
                        "FRAME": filename,
                        "POLYP ID": meta["ID"],
                        "LOCATION": meta.get("LOCATION"),
                        "SHAPE": meta.get("SHAPE"),
                        "SIZE": meta.get("SIZE"),
                        "BOUNDING BOX": bbox,
                    }
                )
        return case_rows

    annotation_args: list[tuple[str, pd.Series]] = []
    for _, row in metadata_df.iterrows():
        case_id = row["ID"]
        txt_file = os.path.join(annotation_txt_dir, f"case{case_id}.txt")
        if os.path.exists(txt_file):
            annotation_args.append((txt_file, row))

    with ThreadPoolExecutor() as executor:
        parsed_rows = executor.map(_parse_case_annotation, annotation_args)
    rows = [r for case_rows in parsed_rows for r in case_rows]

    df = pd.DataFrame(rows)

    df["SRC_PATH"] = df.apply(
        lambda r: os.path.join(
            src_dir, "sundatabase_positive", f"case{r['POLYP ID']}", r["FRAME"]
        ),
        axis=1,
    )

    with ThreadPoolExecutor() as executor:
        sizes = list(executor.map(get_image_size, df["SRC_PATH"]))
    df["WIDTH"], df["HEIGHT"] = zip(*sizes)

    df = _drop_duplicate_frames(df)
    df["BOUNDING BOX"] = _clip_bboxes(df["BOUNDING BOX"], df["WIDTH"], df["HEIGHT"])
    df["IMAGE TYPE"] = "WL"

    out_dir = os.path.join(dest_dir, "sun")
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
    logger.info("Finished SUN conversion with %d entries", len(df))
