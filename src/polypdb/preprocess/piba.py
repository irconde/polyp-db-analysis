"""Utilities for preprocessing the PIBA dataset.

The routine mirrors the operations performed in
``piba-database-analysis.ipynb`` including column renaming, normalisation of
image modality, mapping Paris classification codes and building bounding boxes
from ``(x, y, w, h)`` coordinates. Image dimensions are also recorded so the
resulting ``annotations.xlsx`` closely matches the notebook output.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import logging

from ..utils import get_image_size, move_images_parallel, validate_conversion


def _fetch_metadata_from_db(db_config: dict) -> pd.DataFrame:
    """Return PIBA metadata by executing the SQL query used in the notebook.

    Parameters
    ----------
    db_config:
        Dictionary with ``user``, ``password``, ``host`` and ``database`` keys
        understood by :func:`mysql.connector.connect`.

    Returns
    -------
    pandas.DataFrame
        DataFrame mirroring the ``piba_db_dump.xlsx`` produced in the
        ``piba-database-analysis.ipynb`` notebook.
    """

    try:  # Import lazily so ``mysql-connector-python`` remains optional.
        import mysql.connector  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when module missing
        raise ImportError(
            "mysql-connector-python is required to fetch PIBA metadata"
        ) from exc

    query = (
        "SELECT\n"
        "    img.id AS image_id,\n"
        "    img.manually_selected AS manually_selected,\n"
        "    g.title AS title,\n"
        # Treat polyp id as the case-level identifier
        "    p.id AS case_id,\n"
        "    p.size AS polyp_size,\n"
        "    p.paris_primary AS paris_primary,\n"
        "    p.location AS polyp_location,\n"
        "    pl.x AS x,\n"
        "    pl.y AS y,\n"
        "    pl.width AS width,\n"
        "    pl.height AS height\n"
        "FROM\n"
        "    image img\n"
        "JOIN\n"
        "    polyp p ON img.polyp_id = p.id\n"
        "JOIN\n"
        "    polyplocation pl ON img.id = pl.image_id\n"
        "JOIN\n"
        "    gallery g ON img.gallery_id = g.id\n"
        "WHERE\n"
        "    img.is_removed = FALSE;"
    )

    conn = mysql.connector.connect(**db_config)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


logger = logging.getLogger(__name__)

RENAME_COLUMNS = {
    "image_id": "FRAME",
    "title": "IMAGE TYPE",
    "manually_selected": "MANUAL SELECTION",
    # Support both legacy 'polyp_id' (from Excel dumps) and the explicit
    # 'case_id' alias used when pulling directly from the DB.
    "polyp_id": "POLYP ID",
    "case_id": "POLYP ID",
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
    src_dir: Path | str,
    dest_dir: Path | str,
    metadata_filename: str = "piba_db_dump.xlsx",
    db_config: dict | None = None,
) -> None:
    """Convert PIBA dataset to the standard directory layout.

    Apart from renaming columns, the function normalises the light source
    labels, maps morphology descriptions to Paris codes, computes image
    dimensions and produces bounding boxes in ``x1,y1,x2,y2`` format. When an
    Excel dump of the metadata is not available the information can be fetched
    directly from a MySQL database by passing ``db_config``.

    Args:
        src_dir: Directory containing the original PIBA files.
        dest_dir: Directory where the converted dataset will be written.
        metadata_filename: Name of the Excel file with polyp metadata.
        db_config: Optional connection parameters for the PIBA MySQL database.

    Returns:
        None
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    logger.info("Converting PIBA dataset from %s to %s", src_dir, dest_dir)
    meta_path = src_dir / metadata_filename
    if meta_path.exists():
        df = pd.read_excel(meta_path)
    elif db_config is not None:
        df = _fetch_metadata_from_db(db_config)
    else:
        raise FileNotFoundError(
            f"Metadata file '{metadata_filename}' not found and no db_config provided"
        )
    df = df.rename(columns=RENAME_COLUMNS)
    df["IMAGE TYPE"] = np.where(
        df["IMAGE TYPE"].str.contains("WL"), "WL", "NBI")
    df["SIZE"] = df["SIZE"].astype("Int64")
    df = df.dropna(subset=["SHAPE"])
    df["SHAPE"] = df["SHAPE"].replace(SHAPE_MAP)
    df["FRAME"] = df["FRAME"].apply(lambda x: f"{x}.png")

    def _size(frame: str) -> tuple[int, int]:
        return get_image_size(src_dir / "media" / frame)

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

    out_dir = dest_dir / "piba"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_dir / "annotations.xlsx", index=False)
    move_images_parallel(
        src_dir / "media",
        out_dir / "img",
        df,
        copy=True,
    )
    validate_conversion(out_dir, df)
    logger.info("Finished PIBA conversion with %d entries", len(df))
