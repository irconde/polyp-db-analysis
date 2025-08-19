import os
import pandas as pd
import numpy as np

from ..utils import get_image_size, move_images

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


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert PIBA dataset to the standard directory layout."""
    df = pd.read_excel(os.path.join(src_dir, "piba_db_dump.xlsx"))
    df = df.rename(columns=RENAME_COLUMNS)
    df["IMAGE TYPE"] = np.where(df["IMAGE TYPE"].str.contains("WL"), "WL", "NBI")
    df["SIZE"] = df["SIZE"].astype("Int64")
    df = df.dropna(subset=["SHAPE"])
    df["SHAPE"] = df["SHAPE"].replace(SHAPE_MAP)
    df["FRAME"] = df["FRAME"].apply(lambda x: f"{x}.png")

    widths = []
    heights = []
    for frame in df["FRAME"]:
        w, h = get_image_size(os.path.join(src_dir, "media", frame))
        widths.append(w)
        heights.append(h)
    df["WIDTH"] = widths
    df["HEIGHT"] = heights

    df["BOUNDING BOX"] = df.apply(
        lambda r: f"{r['X']},{r['Y']},{r['X'] + r['BBOX WIDTH']},{r['Y'] + r['BBOX HEIGHT']}",
        axis=1,
    )
    df = df.drop(columns=["X", "Y", "BBOX WIDTH", "BBOX HEIGHT", "MANUAL SELECTION"])

    out_dir = os.path.join(dest_dir, "piba")
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    move_images(os.path.join(src_dir, "media"), os.path.join(out_dir, "img"), df)
