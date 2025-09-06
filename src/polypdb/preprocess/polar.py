"""Utilities for preprocessing the POLAR dataset.

The routine mirrors the operations performed in
``polar-database-analysis.ipynb`` including column renaming, cleaning of
morphology labels, adjustment of bounding boxes that overflow image
boundaries and recording of image dimensions.  The goal is to produce an
``annotations.xlsx`` and image directory that match the structure used by
other datasets in the project.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..utils import get_image_size, move_images_parallel, validate_conversion


logger = logging.getLogger(__name__)

# Column renaming performed prior to normalisation
RENAME_COLUMNS = {
    "POLYP/IMAGE_ID": "FRAME",
    "POLYP SIZE": "SIZE",
    "POLYP MORPHOLOGY": "SHAPE",
    "POLYP LOCATION": "LOCATION",
    "POLYP POSITION (x,y,w,h)": "BOUNDING BOX",
}

# Values that represent missing or unusable morphology descriptions
INVALID_SHAPES = {"##USER_MISSING_97##", "##USER_MISSING_99##", "Stenosing"}

# Map of descriptive morphology labels to Paris classification codes
SHAPE_MAP = {
    "Sessile": "Is",
    "Flat elevated": "IIa",
    "Pedunculated": "Ip",
    "Flat": "IIb",
    "Depressed": "IIc",
}


def convert_to_standard_layout(
    src_dir: Path | str, dest_dir: Path | str, metadata_filename: str = "polar_annotations.xlsx"
) -> None:
    """Convert POLAR dataset to the standard directory layout.

    The function loads the metadata from ``metadata_filename``, applies
    the cleaning steps performed in the analysis notebook and saves the
    resulting annotations and images under ``dest_dir/polar``.

    Args:
        src_dir: Directory containing the original POLAR dataset files.
        dest_dir: Directory where the converted dataset will be written.
        metadata_filename: Name of the Excel file with polyp metadata.

    Returns:
        None
    """

    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    logger.info("Converting POLAR dataset from %s to %s", src_dir, dest_dir)
    df = pd.read_excel(
        src_dir / metadata_filename,
        sheet_name="validation_set",
    )

    # Note: Morphology information is only in the validation_set tab of the spreadsheet
    validation_dir = src_dir / "validation_set"

    df = df.rename(columns=RENAME_COLUMNS)

    # Drop entries with negative sizes and invalid morphology labels
    df = df[df["SIZE"] >= 0]
    df = df[~df["SHAPE"].isin(INVALID_SHAPES)]

    # Map morphology descriptions to Paris codes
    df["SHAPE"] = df["SHAPE"].replace(SHAPE_MAP)

    # Preserve polyp identifier before renaming the image id to FRAME
    df["POLYP ID"] = df["FRAME"].astype(str)

    # Determine image dimensions for each frame. Images are organised in
    # sub-folders per patient, so construct the path accordingly.
    df["SRC_PATH"] = [
        validation_dir / str(pid) / f"{polyp_id}.png"
        for pid, polyp_id in zip(df["PATIENT_ID"], df["POLYP ID"].astype(str))
    ]

    with ThreadPoolExecutor() as executor:
        sizes = list(executor.map(get_image_size, df["SRC_PATH"]))
    df["WIDTH"], df["HEIGHT"] = zip(*sizes)

    # Append the image extension after dimensions are collected
    df["FRAME"] = df["POLYP ID"].astype(str) + ".png"
    df["IMAGE TYPE"] = "NBI"

    # Parse bounding box values and adjust them to stay inside image bounds
    coords = df["BOUNDING BOX"].str.strip(
        "[]").str.split(",", expand=True).astype(int)
    df["x"] = coords[0]
    df["y"] = coords[1]
    df["w"] = coords[2]
    df["h"] = coords[3]

    # Clamp width/height so the box does not overflow the image
    df["w"] = np.minimum(df["w"], df["WIDTH"] - df["x"])
    df["h"] = np.minimum(df["h"], df["HEIGHT"] - df["y"])
    df = df[(df["x"] >= 0) & (df["y"] >= 0) & (df["w"] >= 0) & (df["h"] >= 0)]

    bbox_strings = (
        df.assign(x2=df["x"] + df["w"], y2=df["y"] +
                  df["h"])[["x", "y", "x2", "y2"]]
        .astype(int)
        .astype(str)
        .agg(",".join, axis=1)
    )
    df = df.assign(**{"BOUNDING BOX": bbox_strings})

    out_dir = dest_dir / "polar"
    out_img_dir = out_dir / "img"
    move_images_parallel(
        None,
        out_img_dir,
        df,
        filename_col="FRAME",
        source_path_col="SRC_PATH",
        copy=True,
    )

    df = df.drop(
        columns=[
            "PATIENT_ID",
            "OPTICAL DIAGNOSIS",
            "OPTCAL DIAGNOSIS CONFIDENCE",
            "PATHOLOGY DIAGNOSIS",
            "x",
            "y",
            "w",
            "h",
            "SRC_PATH",
        ],
        errors="ignore",
    )

    df.to_excel(out_dir / "annotations.xlsx", index=False)
    validate_conversion(out_dir, df)
    logger.info("Finished POLAR conversion with %d entries", len(df))
