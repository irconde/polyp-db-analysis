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

from .base import DatasetConverter
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

class PolarConverter(DatasetConverter):
    """Converter for the POLAR dataset."""

    def __init__(
        self,
        src_dir: Path | str,
        dest_dir: Path | str,
        metadata_filename: str = "polar_annotations.xlsx",
    ) -> None:
        super().__init__(src_dir, dest_dir, metadata_filename=metadata_filename)
        self.metadata_filename = metadata_filename
        self.validation_dir = self.src_dir / "validation_set"

    def load_metadata(self) -> pd.DataFrame:
        return pd.read_excel(self.src_dir / self.metadata_filename, sheet_name="validation_set")

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=RENAME_COLUMNS)
        df = df[df["SIZE"] >= 0]
        df = df[~df["SHAPE"].isin(INVALID_SHAPES)]
        df["SHAPE"] = df["SHAPE"].replace(SHAPE_MAP)
        df["POLYP ID"] = df["FRAME"].astype(str)
        df["SRC_PATH"] = [
            self.validation_dir / str(pid) / f"{polyp_id}.png"
            for pid, polyp_id in zip(df["PATIENT_ID"], df["POLYP ID"].astype(str))
        ]
        return df

    def compute_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            sizes = list(executor.map(get_image_size, df["SRC_PATH"]))
        df["WIDTH"], df["HEIGHT"] = zip(*sizes)

        df["FRAME"] = df["POLYP ID"].astype(str) + ".png"
        df["IMAGE TYPE"] = "NBI"

        coords = df["BOUNDING BOX"].str.strip("[]").str.split(",", expand=True).astype(int)
        df["x"] = coords[0]
        df["y"] = coords[1]
        df["w"] = coords[2]
        df["h"] = coords[3]

        df["w"] = np.minimum(df["w"], df["WIDTH"] - df["x"])
        df["h"] = np.minimum(df["h"], df["HEIGHT"] - df["y"])
        df = df[(df["x"] >= 0) & (df["y"] >= 0) & (df["w"] >= 0) & (df["h"] >= 0)]

        bbox_strings = (
            df.assign(x2=df["x"] + df["w"], y2=df["y"] + df["h"])[["x", "y", "x2", "y2"]]
            .astype(int)
            .astype(str)
            .agg(",".join, axis=1)
        )
        df = df.assign(**{"BOUNDING BOX": bbox_strings})
        return df

    def write_outputs(self, df: pd.DataFrame) -> None:
        out_dir = self.dest_dir / "polar"
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


def convert_to_standard_layout(
    src_dir: Path | str, dest_dir: Path | str, metadata_filename: str = "polar_annotations.xlsx"
) -> None:
    """Backward compatible wrapper around :class:`PolarConverter`."""

    converter = PolarConverter(src_dir, dest_dir, metadata_filename=metadata_filename)
    converter.convert()
