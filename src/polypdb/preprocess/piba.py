"""Utilities for preprocessing the PIBA dataset.

The routine mirrors the operations performed in
``piba-database-analysis.ipynb`` including column renaming, normalisation of
image modality, mapping Paris classification codes and building bounding boxes
from ``(x, y, w, h)`` coordinates. Image dimensions are also recorded so the
resulting ``annotations.xlsx`` closely matches the notebook output.
"""

import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

from ..utils import get_image_size, move_images_parallel, validate_conversion

RENAME_COLUMNS = {
    "image_id": "FRAME",
    "title": "IMAGE TYPE",
    "manually_selected": "MANUAL SELECTION",
    "polyp_id": "POLYP ID",
    "polyp_size": "SIZE",
    "paris_primary": "SHAPE",
    "polyp_location": "LOCATION",
    "width": "BBOX WIDTH",
    "height": "BBOX HEIGHT",
    "x": "X",
    "y": "Y",
}

SHAPE_MAP = {
    "DEPRESSED": "IIc",
    "FLAT": "IIb",
    "PEDUNCULATED": "Ip",
    "SESSILE": "Is",
    "SLIGHTLY_ELEVATED": "IIa",
    "ULCERATED": "III",
}


def convert_to_standard_layout(
    src_dir: str, dest_dir: str, metadata_filename: str = "piba_db_dump.xlsx"
) -> None:
    """Convert PIBA dataset to the standard directory layout.

    Apart from renaming columns, the function normalises the light source
    labels, maps morphology descriptions to Paris codes, computes image
    dimensions and produces bounding boxes in ``x1,y1,x2,y2`` format.

    Args:
        src_dir: Directory containing the original PIBA files.
        dest_dir: Directory where the converted dataset will be written.
        metadata_filename: Name of the Excel file with polyp metadata.

    Returns:
        None
    """
    df = pd.read_excel(os.path.join(src_dir, metadata_filename))
    df = df.rename(columns=RENAME_COLUMNS)
    df["IMAGE TYPE"] = np.where(df["IMAGE TYPE"].str.contains("WL"), "WL", "NBI")
    df["SIZE"] = df["SIZE"].astype("Int64")
    df = df.dropna(subset=["SHAPE"])
    df["SHAPE"] = df["SHAPE"].replace(SHAPE_MAP)
    df["FRAME"] = df["FRAME"].apply(lambda x: f"{x}.png")

    def _size(frame: str) -> tuple[int, int]:
        return get_image_size(os.path.join(src_dir, "media", frame))

    with ThreadPoolExecutor() as executor:
        sizes = list(executor.map(_size, df["FRAME"]))
    df["WIDTH"], df["HEIGHT"] = zip(*sizes)
    bbox_strings = (
        df.assign(
            x2=df["X"] + df["BBOX WIDTH"],
            y2=df["Y"] + df["BBOX HEIGHT"],
        )[["X", "Y", "x2", "y2"]]
        .astype("Int64")
        .astype(str)
        .agg(",".join, axis=1)
    )
    df = df.assign(**{"BOUNDING BOX": bbox_strings}).drop(
        columns=["X", "Y", "BBOX WIDTH", "BBOX HEIGHT", "MANUAL SELECTION"]
    )

    out_dir = os.path.join(dest_dir, "piba")
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    move_images_parallel(
        os.path.join(src_dir, "media"), os.path.join(out_dir, "img"), df
    )
    validate_conversion(out_dir, df)
