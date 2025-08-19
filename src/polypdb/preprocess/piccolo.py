import os
import pandas as pd

from ..utils import move_images


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert PICCOLO dataset to the standard directory layout."""
    df = pd.read_csv(os.path.join(src_dir, "annotations.csv"))
    df = df.rename(
        columns={
            "POLYP SIZE (mm)": "SIZE",
            "PARIS CLASSIFICATION": "SHAPE",
            "CODE - LESION": "POLYP ID",
        }
    )
    df["FRAME"] = df["FRAME"].astype(str)
    df["FRAME"] = df["FRAME"].apply(lambda x: x if x.lower().endswith(".png") else f"{x}.png")

    out_dir = os.path.join(dest_dir, "piccolo")
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    move_images(os.path.join(src_dir, "cropped_polyps"), os.path.join(out_dir, "img"), df)
