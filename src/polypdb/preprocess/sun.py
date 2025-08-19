import os
import pandas as pd

from ..utils import move_images


def convert_to_standard_layout(src_dir: str, dest_dir: str) -> None:
    """Convert SUN dataset to the standard directory layout."""
    df = pd.read_excel(os.path.join(src_dir, "annotations.xlsx"))
    df["FRAME"] = df["FRAME"].astype(str)
    df["FRAME"] = df["FRAME"].apply(lambda x: x if x.lower().endswith(".png") else f"{x}.png")
    df["IMAGE TYPE"] = "WL"

    out_dir = os.path.join(dest_dir, "sun")
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    move_images(os.path.join(src_dir, "img_all"), os.path.join(out_dir, "img"), df)
