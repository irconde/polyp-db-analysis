from __future__ import annotations

"""Utilities for preprocessing the PolypGen dataset.

This module processes the single-frame data provided as part of the
PolypGen2021 multi-centre dataset.  Each centre directory contains the
``images_CN`` folder with image frames and a matching ``bbox_CN`` directory
with one text file per image describing the bounding boxes in a simple
VOC-like format.  Empty annotation files indicate that no polyps are present in
that frame.

The :func:`convert_to_standard_layout` function converts these files into the
canonical layout used throughout the project, copying the images and producing
an ``annotations.xlsx`` file with normalised bounding box coordinates.
"""

import logging
from pathlib import Path
import re

import cv2
import pandas as pd

from .base import DatasetConverter
from ..utils import crop_black_padding_band, validate_conversion
from ..polypgen_viewport import mask_small_viewport

logger = logging.getLogger(__name__)


# Frames with known issues that should be excluded from processing
EXCLUDED_FRAMES: dict[str, set[str]] = {
    "C1": {
        "356OLCV1_100S0007"
    },
    "C2": {
        "EndoCV2021_001010",
        "EndoCV2021_001024",
        "EndoCV2021_001025",
        "EndoCV2021_001028",
        "EndoCV2021_001029",
        "EndoCV2021_001030",
        "EndoCV2021_001031",
        "EndoCV2021_001036",
        "EndoCV2021_001043",
        "EndoCV2021_001044",
        "EndoCV2021_001106",
        "EndoCV2021_001107",
        "EndoCV2021_001108",
        "EndoCV2021_001117",
        "EndoCV2021_001118",
        "EndoCV2021_001149",
        "EndoCV2021_001150",
        "EndoCV2021_001166",
        "EndoCV2021_001215",
        "EndoCV2021_001216",
        "EndoCV2021_001217",
        "EndoCV2021_001218",
        "EndoCV2021_001273",
        "EndoCV2021_001274",
        "EndoCV2021_001292",
        "EndoCV2021_001315",
        "EndoCV2021_001316",
        "EndoCV2021_001317",
        "EndoCV2021_001318",
        "EndoCV2021_001319",
        "EndoCV2021_001320",
        "EndoCV2021_001336",
        "EndoCV2021_001337",
        "EndoCV2021_001338",
        "EndoCV2021_001191",
        "EndoCV2021_001260",
        "EndoCV2021_001282",
        "EndoCV2021_001321"
    },
    "C3": {
        "C3_EndoCV2021_00480",
        "C3_EndoCV2021_00481",
        "C3_EndoCV2021_00482",
        "C3_EndoCV2021_00491",
    },
    "C5": {
        "EndoCV2021_C5_50000057",
        "EndoCV2021_C5_50000314",
        "EndoCV2021_C5_50000100",
        "EndoCV2021_C5_50000101",
        "EndoCV2021_C5_50000102",
        "EndoCV2021_C5_50000103",
        "EndoCV2021_C5_50000108",
        "EndoCV2021_C5_50000109",
        "EndoCV2021_C5_50000110",
        "EndoCV2021_C5_50000111",
        "EndoCV2021_C5_50000112",
        "EndoCV2021_C5_50000113",
        "EndoCV2021_C5_50000115",
        "EndoCV2021_C5_50000116",
        "EndoCV2021_C5_50000120",
        "EndoCV2021_C5_50000121",
        "EndoCV2021_C5_50000122",
        "EndoCV2021_C5_50000123",
        "EndoCV2021_C5_50000124",
        "EndoCV2021_C5_50000125",
        "EndoCV2021_C5_50000126",
        "EndoCV2021_C5_50000127",
        "EndoCV2021_C5_50000128",
        "EndoCV2021_C5_50000129",
        "EndoCV2021_C5_50000138",
        "EndoCV2021_C5_50000139",
        "EndoCV2021_C5_50000140",
        "EndoCV2021_C5_50000141"
    },
    "C6": {
        "EndoCV2021_C6_0100001",
        "EndoCV2021_C6_0100003",
        "EndoCV2021_C6_0100004",
        "EndoCV2021_C6_0100005",
        "EndoCV2021_C6_0100006",
        "EndoCV2021_C6_0100007",
        "EndoCV2021_C6_0100008",
        "EndoCV2021_C6_0100009",
        "EndoCV2021_C6_0100010",
        "EndoCV2021_C6_0100011",
        "EndoCV2021_C6_0100012",
        "EndoCV2021_C6_0100013",
        "EndoCV2021_C6_0100014",
        "EndoCV2021_C6_0100015",
        "EndoCV2021_C6_0100017",
        "EndoCV2021_C6_0100018",
        "EndoCV2021_C6_0100019",
        "EndoCV2021_C6_0100020",
        "EndoCV2021_C6_0100021",
        "EndoCV2021_C6_0100022",
        "EndoCV2021_C6_0100023",
        "EndoCV2021_C6_0100025",
        "EndoCV2021_C6_0100027",
        "EndoCV2021_C6_0100028",
        "EndoCV2021_C6_0100030",
        "EndoCV2021_C6_0100031",
        "EndoCV2021_C6_0100094",
    },
}


def _parse_bbox_file(path: Path | str) -> str:
    """Return bounding boxes from ``path`` normalised to ``x1,y1,x2,y2`` format.

    Lines in the file may optionally start with a class label.  Coordinates are
    interpreted either as ``x_min y_min x_max y_max`` or ``x_min y_min width height``.
    Multiple boxes are joined using spaces.  Missing files or empty annotations
    result in an empty string.
    """

    path = Path(path)
    if not path.exists():
        return ""

    boxes: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
            # Drop non-numeric class label when present
            if parts and not re.match(r"^-?\d+(?:\.\d+)?$", parts[0]):
                parts = parts[1:]
            if len(parts) < 4:
                continue
            nums = list(map(float, parts[:4]))
            x1, y1, x2, y2 = nums
            if x2 <= x1 or y2 <= y1:
                # Interpret as x, y, width, height
                x1, y1, w, h = nums
                x2 = x1 + w
                y2 = y1 + h
            boxes.append(
                f"{int(round(x1))},{int(round(y1))},{int(round(x2))},{int(round(y2))}"
            )
    # Use semicolons to separate multiple boxes to match downstream expectations
    return ";".join(boxes)


class PolypGenConverter(DatasetConverter):
    """Converter for the PolypGen dataset."""

    def __init__(
        self,
        src_dir: Path | str,
        dest_dir: Path | str,
        *,
        save_raw: bool = False,
        skip_cleaning: bool = False,
        **kwargs,
    ) -> None:
        """Create a converter for the PolypGen dataset.

        Parameters
        ----------
        src_dir, dest_dir:
            Source and destination directories.
        save_raw:
            When ``True`` the original, unmodified frames are copied to a
            ``raw_img`` directory and their dimensions/bounding boxes are
            recorded in additional columns.
        skip_cleaning:
            Bypass black band cropping and viewport masking entirely. The
            images written to ``img`` will be identical to the originals.
        """

        super().__init__(src_dir, dest_dir, **kwargs)
        self.save_raw = save_raw
        self.skip_cleaning = skip_cleaning

    def load_metadata(self) -> pd.DataFrame:
        rows: list[dict] = []
        for centre in range(1, 7):
            centre_name = f"C{centre}"
            centre_dir = self.src_dir / f"data_{centre_name}"
            img_dir = centre_dir / f"images_{centre_name}"
            bbox_dir = centre_dir / f"bbox_{centre_name}"
            if not img_dir.is_dir() or not bbox_dir.is_dir():
                continue
            for img_file in sorted(f.name for f in img_dir.iterdir() if f.is_file()):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                base = Path(img_file).stem
                if base in EXCLUDED_FRAMES.get(centre_name, set()):
                    continue
                bbox_file = bbox_dir / f"{base}.txt"
                if not bbox_file.exists():
                    alt_bbox = bbox_dir / f"{base}_mask.txt"
                    if alt_bbox.exists():
                        bbox_file = alt_bbox
                bbox = _parse_bbox_file(bbox_file)
                src_path = img_dir / img_file
                rows.append(
                    {
                        "FRAME": img_file,
                        "POLYP ID": pd.NA,
                        "CENTER": centre_name,
                        "BOUNDING BOX": bbox,
                        "SRC_PATH": src_path,
                        "IMAGE TYPE": "WL",
                    }
                )
        df = pd.DataFrame(rows)
        return df.rename(columns={"BOUNDING BOX": "BOUNDING_BOX"})

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def compute_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        out_dir = self.dest_dir / "polypgen"
        img_dest_dir = out_dir / "img"
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        raw_dest_dir = out_dir / "raw_img"
        raw_widths: list[int] = []
        raw_heights: list[int] = []
        raw_boxes: list[str] = []

        widths: list[int] = []
        heights: list[int] = []
        boxes: list[str] = []

        for row in df.itertuples(index=False):
            img = cv2.imread(str(row.SRC_PATH))
            if img is None:
                continue

            if self.save_raw:
                raw_dest_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(raw_dest_dir / row.FRAME), img)
                rh, rw = img.shape[:2]
                raw_widths.append(rw)
                raw_heights.append(rh)
                raw_boxes.append(row.BOUNDING_BOX or "")

            if self.skip_cleaning:
                cleaned = img
                x_off = y_off = 0
            else:
                cropped, x_off, y_off = crop_black_padding_band(img)
                cleaned, _ = mask_small_viewport(cropped)

            cv2.imwrite(str(img_dest_dir / row.FRAME), cleaned)
            h, w = cleaned.shape[:2]
            widths.append(w)
            heights.append(h)

            bbox_str = row.BOUNDING_BOX
            if bbox_str and not self.skip_cleaning:
                adj_boxes: list[str] = []
                for box in re.split(r"[\s;]+", bbox_str.strip()):
                    x1, y1, x2, y2 = map(int, box.split(","))
                    adj_boxes.append(
                        f"{x1 - x_off},{y1 - y_off},{x2 - x_off},{y2 - y_off}"
                    )
                boxes.append(";".join(adj_boxes))
            else:
                boxes.append(bbox_str or "")

        df["WIDTH"] = widths
        df["HEIGHT"] = heights
        df["BOUNDING_BOX"] = boxes

        if self.save_raw:
            df["RAW_WIDTH"] = raw_widths
            df["RAW_HEIGHT"] = raw_heights
            df["RAW_BOUNDING_BOX"] = raw_boxes

        df = df.drop(columns=["SRC_PATH"], errors="ignore")
        return df

    def write_outputs(self, df: pd.DataFrame) -> None:
        out_dir = getattr(self, "out_dir", self.dest_dir / "polypgen")
        rename_map = {"BOUNDING_BOX": "BOUNDING BOX"}
        if "RAW_BOUNDING_BOX" in df.columns:
            rename_map["RAW_BOUNDING_BOX"] = "RAW BOUNDING BOX"
        if "RAW_WIDTH" in df.columns:
            rename_map["RAW_WIDTH"] = "RAW WIDTH"
        if "RAW_HEIGHT" in df.columns:
            rename_map["RAW_HEIGHT"] = "RAW HEIGHT"
        df = df.rename(columns=rename_map)
        df.to_excel(out_dir / "annotations.xlsx", index=False)
        validate_conversion(out_dir, df)
        logger.info("Finished PolypGen conversion with %d entries", len(df))


def convert_to_standard_layout(
    src_dir: Path | str,
    dest_dir: Path | str,
    *,
    save_raw: bool = False,
    skip_cleaning: bool = False,
) -> None:
    """Backward compatible wrapper around :class:`PolypGenConverter`."""

    converter = PolypGenConverter(
        src_dir, dest_dir, save_raw=save_raw, skip_cleaning=skip_cleaning
    )
    converter.convert()
