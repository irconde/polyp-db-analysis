import os
import pandas as pd

from ..utils import move_images


def _transform_bbox(bbox_str: str) -> str:
    x, y, w, h = map(int, bbox_str.strip('[]').split(','))
    return f"{x},{y},{x + w},{y + h}"


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert POLAR dataset to the standard directory layout."""
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
    df["IMAGE TYPE"] = "NBI"
    df["BOUNDING BOX"] = df["BOUNDING BOX"].apply(_transform_bbox)
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
