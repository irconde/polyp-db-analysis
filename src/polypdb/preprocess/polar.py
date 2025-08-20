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
import pandas as pd

from ..utils import get_image_size, move_images

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


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert POLAR dataset to the standard directory layout.

    The function loads the metadata from ``polar_annotations.xlsx``, applies
    the cleaning steps performed in the analysis notebook and saves the
    resulting annotations and images under ``dest_dir/polar``.
    """

    df = pd.read_excel(
        os.path.join(src_dir, "polar_annotations.xlsx"),
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
    df["FRAME"] = df["FRAME"].astype(str) + ".png"
    df["IMAGE TYPE"] = "NBI"

    # Determine image dimensions for each frame
    widths: list[int] = []
    heights: list[int] = []
    for frame in df["FRAME"]:
        w, h = get_image_size(os.path.join(validation_dir, "img", frame))
        widths.append(w)
        heights.append(h)
    df["WIDTH"] = widths
    df["HEIGHT"] = heights

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

    out_dir = os.path.join(dest_dir, "polar")
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    move_images(os.path.join(src_dir, "img"), os.path.join(out_dir, "img"), df)

