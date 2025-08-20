"""Utilities for preprocessing the POLAR dataset.

The routine mirrors the operations performed in
``polar-database-analysis.ipynb`` including column renaming, cleaning of
morphology labels, adjustment of bounding boxes that overflow image
boundaries and recording of image dimensions.  The goal is to produce an
``annotations.xlsx`` and image directory that match the structure used by
other datasets in the project.
"""

from __future__ import annotations

import os
import shutil
import pandas as pd
import logging

from ..utils import get_image_size


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
    src_dir: str, dest_dir: str, metadata_filename: str = "polar_annotations.xlsx"
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

    df = pd.read_excel(
        os.path.join(src_dir, metadata_filename),
        sheet_name="validation_set",
    )

    # Note: Morphology information is only in the validation_set tab of the spreadsheet
    validation_dir = os.path.join(src_dir, "validation_set")

    df = df.rename(columns=RENAME_COLUMNS)

    # Drop entries with negative sizes and invalid morphology labels
    df = df[df["SIZE"] >= 0]
    df = df[~df["SHAPE"].isin(INVALID_SHAPES)]

    # Map morphology descriptions to Paris codes
    df["SHAPE"] = df["SHAPE"].replace(SHAPE_MAP)

    # Preserve polyp identifier before renaming the image id to FRAME
    df["POLYP ID"] = df["FRAME"].astype(str)

    # Determine image dimensions for each frame.  Images are organised in
    # sub-folders per patient, so construct the path accordingly.
    widths: list[int] = []
    heights: list[int] = []
    for _, row in df.iterrows():
        image_path = os.path.join(
            validation_dir, str(row["PATIENT_ID"]), row["POLYP ID"] + ".png"
        )
        w, h = get_image_size(image_path)
        widths.append(w)
        heights.append(h)
    df["WIDTH"] = widths
    df["HEIGHT"] = heights

    # Append the image extension after dimensions are collected
    df["FRAME"] = df["POLYP ID"].astype(str) + ".png"
    df["IMAGE TYPE"] = "NBI"

    # Parse bounding box values and adjust them to stay inside image bounds
    coords = df["BOUNDING BOX"].str.strip("[]").str.split(",", expand=True).astype(int)
    df["x"] = coords[0]
    df["y"] = coords[1]
    df["w"] = coords[2]
    df["h"] = coords[3]

    # Clamp width/height so the box does not overflow the image
    df["w"] = df.apply(lambda r: min(r["w"], r["WIDTH"] - r["x"]), axis=1)
    df["h"] = df.apply(lambda r: min(r["h"], r["HEIGHT"] - r["y"]), axis=1)
    df = df[(df["x"] >= 0) & (df["y"] >= 0) & (df["w"] >= 0) & (df["h"] >= 0)]

    df["BOUNDING BOX"] = df.apply(
        lambda r: f"{r['x']},{r['y']},{r['x'] + r['w']},{r['y'] + r['h']}",
        axis=1,
    )

    out_dir = os.path.join(dest_dir, "polar")
    out_img_dir = os.path.join(out_dir, "img")
    os.makedirs(out_img_dir, exist_ok=True)

    # Move images into the destination directory, flattening the original
    # patient-based structure.
    for _, row in df.iterrows():
        src_img = os.path.join(
            validation_dir, str(row["PATIENT_ID"]), row["POLYP ID"] + ".png"
        )
        dst_img = os.path.join(out_img_dir, row["FRAME"])
        try:
            shutil.move(src_img, dst_img)
        except OSError as e:
            logger.error("Could not move %s. Error: %s", src_img, e)

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
        ],
        errors="ignore",
    )

    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)

