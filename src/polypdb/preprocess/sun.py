"""Preprocessing helpers for the SUN dataset.

The original SUN distribution provides a spreadsheet describing each polyp
and a set of text files with the bounding-box annotations.  The exploratory
``sun-database-analysis.ipynb`` notebook performs a fair amount of munging on
this data: normalising column names, fixing shape labels, parsing the text
annotations, clipping bounding boxes that extend outside the image and
recording the size of every frame.  The goal of this module is to replicate
that logic so that the dataset can be converted automatically without having
to rely on the notebook.
"""

import os
import shutil
import pandas as pd

from ..utils import get_image_size


def _clip_bbox(bbox: str, width: int, height: int) -> str:
    """Clip ``bbox`` coordinates so that they lie within the image bounds.

    Args:
        bbox: One or more bounding boxes separated by spaces in
            ``x1,y1,x2,y2`` format.
        width: Width of the image used for clipping.
        height: Height of the image used for clipping.

    Returns:
        A space-separated string of bounding boxes with coordinates clamped
        to the image size.
    """

    boxes = []
    for bb in bbox.split(" "):
        if not bb:
            continue
        x1, y1, x2, y2 = map(int, bb.split(","))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        boxes.append(f"{x1},{y1},{x2},{y2}")
    return " ".join(boxes)


def _extract_bboxes(line: str) -> tuple[str, str]:
    """Return the filename and bounding boxes encoded in ``line``.

    The annotation files contain one line per image where the first token is
    the image filename followed by ``x1,y1,x2,y2,0`` entries. The trailing
    ``0`` represents the class id (polyp) and is discarded.

    Args:
        line: Raw line from an annotation text file.

    Returns:
        A tuple containing the image filename and a space-separated string of
        bounding boxes.
    """

    parts = line.strip().split(" ")
    filename = parts[0]
    bbox_strings = []
    for bbox_str in parts[1:]:
        bbox_values = bbox_str.split(",")[:-1]  # drop class id
        bbox_strings.append(",".join(bbox_values))
    return filename, " ".join(bbox_strings)


def convert_to_standard_layout(
    src_dir: str, dest_dir: str, metadata_filename: str = "metadata.csv"
) -> None:
    """Convert SUN dataset to the standard directory layout.

    ``src_dir`` should contain the ``metadata_filename`` file as well as the
    ``sundatabase_positive`` folder with ``annotation_txt`` and ``case*``
    subdirectories. The resulting annotations and images are written under
    ``dest_dir/sun``.

    Args:
        src_dir: Root directory of the SUN dataset.
        dest_dir: Directory where the converted dataset will be written.
        metadata_filename: Name of the CSV file with polyp metadata.

    Returns:
        None
    """

    metadata_path = os.path.join(src_dir, metadata_filename)
    df = pd.read_csv(metadata_path)
    df.columns = [c.upper() for c in df.columns]
    df["SHAPE"] = df["SHAPE"].replace("IIa(LST-NG)", "IIa")
    df = df.drop(columns=["PATHOLOGICAL DIAGNOSIS"], errors="ignore")

    annotation_txt_dir = os.path.join(src_dir, "sundatabase_positive", "annotation_txt")
    rows = []
    for _, row in df.iterrows():
        case_id = row["ID"]
        txt_file = os.path.join(annotation_txt_dir, f"case{case_id}.txt")
        if not os.path.exists(txt_file):
            continue
        with open(txt_file, "r") as f:
            for line in f:
                filename, bbox = _extract_bboxes(line)
                rows.append(
                    {
                        "FRAME": filename,
                        "POLYP ID": case_id,
                        "LOCATION": row.get("LOCATION"),
                        "SHAPE": row.get("SHAPE"),
                        "SIZE": row.get("SIZE"),
                        "BOUNDING BOX": bbox,
                    }
                )

    df = pd.DataFrame(rows)

    widths = []
    heights = []
    for _, r in df.iterrows():
        img_path = os.path.join(
            src_dir, "sundatabase_positive", f"case{r['POLYP ID']}", r["FRAME"]
        )
        w, h = get_image_size(img_path)
        widths.append(w)
        heights.append(h)
    df["WIDTH"] = widths
    df["HEIGHT"] = heights

    df["BOUNDING BOX"] = df.apply(
        lambda r: _clip_bbox(r["BOUNDING BOX"], r["WIDTH"], r["HEIGHT"]), axis=1
    )
    df["IMAGE TYPE"] = "WL"

    out_dir = os.path.join(dest_dir, "sun")
    img_dest_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dest_dir, exist_ok=True)

    for _, r in df.iterrows():
        src_img = os.path.join(
            src_dir, "sundatabase_positive", f"case{r['POLYP ID']}", r["FRAME"]
        )
        dst_img = os.path.join(img_dest_dir, r["FRAME"])
        try:
            shutil.copy(src_img, dst_img)
        except IOError as e:
            print(f"Could not copy {r['FRAME']}. Error: {e}")

    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
