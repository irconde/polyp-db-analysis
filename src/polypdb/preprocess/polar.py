import os
import pandas as pd

from ..utils import get_image_size, move_images


def _transform_bbox(bbox_str: str) -> str:
    x, y, w, h = map(int, bbox_str.strip('[]').split(','))
    return f"{x},{y},{x + w},{y + h}"


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert POLAR dataset to the standard directory layout.

    In addition to the basic column renaming performed by the notebooks, this
    implementation drops entries with negative polyp sizes, converts bounding
    boxes from ``(x, y, w, h)`` to ``(x1, y1, x2, y2)`` coordinates and records
    the size of every image.  These steps mirror the data munging shown in
    ``polar-database-analysis.ipynb``.
    """
    df = pd.read_excel(os.path.join(src_dir, "annotations.xlsx"), sheet_name="validation_set")
    df = df.rename(
        columns={
            "POLYP/IMAGE_ID": "FRAME",
            "POLYP SIZE": "SIZE",
            "POLYP MORPHOLOGY": "SHAPE",
            "POLYP LOCATION": "LOCATION",
        }
    )
    df["FRAME"] = df["FRAME"].astype(str) + ".png"
    df = df[df["SIZE"] >= 0]
    df["IMAGE TYPE"] = "NBI"
    df["BOUNDING BOX"] = df["BOUNDING BOX"].apply(_transform_bbox)

    widths = []
    heights = []
    for frame in df["FRAME"]:
        w, h = get_image_size(os.path.join(src_dir, "img", frame))
        widths.append(w)
        heights.append(h)
    df["WIDTH"] = widths
    df["HEIGHT"] = heights
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
