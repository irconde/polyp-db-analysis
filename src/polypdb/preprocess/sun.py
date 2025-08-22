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
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity

from ..utils import get_image_size, move_images_parallel, validate_conversion


logger = logging.getLogger(__name__)


def _pixel_hash(path: str) -> str:
    """Return a hash of the RGB pixel data found at ``path``."""
    with Image.open(path).convert("RGB") as im:
        arr = np.asarray(im)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _perceptual_hash(
    path: str, hash_func, hash_type: str, hash_size: int | None = None
) -> str:
    """Return perceptual hash of image at ``path`` using ``hash_func``.

    ``hash_size`` is forwarded to ``hash_func`` when provided. For the ``phash``
    algorithm the default hash size is increased to 16 to improve sensitivity,
    unless an explicit ``hash_size`` is supplied.
    """
    with Image.open(path).convert("RGB") as im:
        kwargs = {}
        if hash_type == "phash":
            kwargs["hash_size"] = hash_size if hash_size is not None else 16
        elif hash_size is not None:
            kwargs["hash_size"] = hash_size
        try:
            return str(hash_func(im, **kwargs))
        except TypeError:
            return str(hash_func(im))


def _ssim(path1: str, path2: str) -> float:
    """Return structural similarity between images at ``path1`` and ``path2``."""
    with Image.open(path1).convert("RGB") as im1, Image.open(path2).convert("RGB") as im2:
        arr1 = np.asarray(im1)
        arr2 = np.asarray(im2)
    return structural_similarity(arr1, arr2, channel_axis=2)


LOSSLESS_EXTS = {".png", ".bmp", ".tiff", ".tif"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def _rank_key(path: str, width: int, height: int, filesize: int) -> tuple:
    """Ranking key used to pick the canonical frame within duplicates."""
    ext = os.path.splitext(path)[1].lower()
    is_lossy = 1 if ext not in LOSSLESS_EXTS else 0
    return (-height * width, is_lossy, -filesize, path)


def _drop_duplicate_frames(
    df: pd.DataFrame,
    cache_file: str | None = None,
    hash_type: str = "phash",
    hash_size: int = 16,
    hamming_threshold: int = 4,
    use_ssim: bool = False,
    ssim_threshold: float = 0.99,
) -> pd.DataFrame:
    """Return ``df`` without duplicate frames based on pixel or perceptual content.

    If ``cache_file`` is provided, hashes and file sizes are stored in and
    loaded from this cache to avoid recomputing them on subsequent runs.
    ``hash_size`` controls the size used for perceptual hashing (16 for
    ``phash`` by default) while ``hamming_threshold`` determines the maximum
    distance for near-duplicate detection.
    """

    if df.empty:
        return df

    df = df.copy()

    try:
        hash_func = getattr(imagehash, hash_type)
    except AttributeError as exc:
        raise ValueError(f"Unknown hash type: {hash_type}") from exc

    cache_cols = ["SRC_PATH", "_pixel_hash", "_perceptual_hash", "_filesize", "_drop"]
    if cache_file and os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file)
            if "_perceptual_hash" in cache_df.columns:
                cache_df["_perceptual_hash"] = cache_df["_perceptual_hash"].astype(str)
        except Exception:
            cache_df = pd.DataFrame(columns=cache_cols)
    else:
        cache_df = pd.DataFrame(columns=cache_cols)

    for col in cache_cols:
        if col not in cache_df.columns:
            cache_df[col] = pd.Series(dtype=object)
    cache_df["_drop"] = cache_df["_drop"].fillna(False).astype(bool)

    rows_to_update: list[dict] = []
    drop_paths: set[str] = set()
    for path in df["SRC_PATH"]:
        filesize = os.path.getsize(path)
        cached = cache_df[cache_df["SRC_PATH"] == path]
        if (
            not cached.empty
            and bool(cached["_drop"].iloc[0])
            and int(cached["_filesize"].iloc[0]) == filesize
        ):
            drop_paths.add(path)
            continue
        needs_update = (
            cached.empty
            or int(cached["_filesize"].iloc[0]) != filesize
            or pd.isna(cached["_perceptual_hash"].iloc[0])
        )
        if needs_update:
            rows_to_update.append(
                {
                    "SRC_PATH": path,
                    "_pixel_hash": _pixel_hash(path),
                    "_perceptual_hash": _perceptual_hash(
                        path, hash_func, hash_type, hash_size
                    ),
                    "_filesize": filesize,
                    "_drop": False,
                }
            )

    if drop_paths:
        df = df[~df["SRC_PATH"].isin(drop_paths)]

    if rows_to_update:
        update_df = pd.DataFrame(rows_to_update)
        cache_df = pd.concat(
            [cache_df[~cache_df["SRC_PATH"].isin(update_df["SRC_PATH"])], update_df],
            ignore_index=True,
        )
        if cache_file:
            cache_df.to_csv(cache_file, index=False)

    df = df.merge(cache_df, on="SRC_PATH", how="left")

    if not rows_to_update:
        return df.drop(columns=["_pixel_hash", "_perceptual_hash", "_filesize", "_drop"])

    # Collapse exact pixel duplicates first
    duplicates: list[str] = []
    for _, group in df.groupby("_pixel_hash"):
        if len(group) <= 1:
            continue
        ranked = sorted(
            group.itertuples(),
            key=lambda r: _rank_key(
                r.SRC_PATH,
                r.WIDTH,
                r.HEIGHT,
                getattr(r, "_filesize", os.path.getsize(r.SRC_PATH)),
            ),
        )
        duplicates.extend(r.SRC_PATH for r in ranked[1:])

    if duplicates:
        logger.info("Omitting %d duplicate frames", len(duplicates))
        df = df[~df["SRC_PATH"].isin(duplicates)]
        cache_df.loc[cache_df["SRC_PATH"].isin(duplicates), "_drop"] = True

    # Cluster by perceptual hash for near duplicates
    if hamming_threshold is not None and hamming_threshold >= 0:
        indices = df.index.tolist()
        hashes = [imagehash.hex_to_hash(h) for h in df["_perceptual_hash"]]
        parent = {idx: idx for idx in indices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if (
                    df.at[indices[i], "WIDTH"] != df.at[indices[j], "WIDTH"]
                    or df.at[indices[i], "HEIGHT"] != df.at[indices[j], "HEIGHT"]
                ):
                    continue
                if hashes[i] - hashes[j] <= hamming_threshold:
                    if not use_ssim or _ssim(
                        df.at[indices[i], "SRC_PATH"],
                        df.at[indices[j], "SRC_PATH"],
                    ) >= ssim_threshold:
                        union(indices[i], indices[j])

        clusters: dict[int, list[int]] = {}
        for idx in indices:
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        near_dupes: list[str] = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) <= 1:
                continue
            group = df.loc[cluster_indices]
            ranked = sorted(
                group.itertuples(),
                key=lambda r: _rank_key(
                    r.SRC_PATH,
                    r.WIDTH,
                    r.HEIGHT,
                    getattr(r, "_filesize", os.path.getsize(r.SRC_PATH)),
                ),
            )
            near_dupes.extend(r.SRC_PATH for r in ranked[1:])

        if near_dupes:
            logger.info("Omitting %d near-duplicate frames", len(near_dupes))
            df = df[~df["SRC_PATH"].isin(near_dupes)]
            cache_df.loc[cache_df["SRC_PATH"].isin(near_dupes), "_drop"] = True

    if cache_file:
        cache_df.to_csv(cache_file, index=False)

    return df.drop(columns=["_pixel_hash", "_perceptual_hash", "_filesize", "_drop"])


def drop_duplicate_negative_images(
    src_dir: str,
    cache_file: str | None = None,
    hash_type: str = "phash",
    hash_size: int = 16,
    hamming_threshold: int = 4,
    use_ssim: bool = False,
    ssim_threshold: float = 0.99,
) -> int:
    """Remove duplicate negative images in ``src_dir``.

    The directory tree is traversed recursively and images with identical or
    near-identical content are deleted in-place using :func:`_drop_duplicate_frames`.

    Parameters
    ----------
    src_dir:
        Root directory containing negative images.
    cache_file:
        Optional CSV cache path used to store hashes. When ``None`` a cache
        file named ``sun_neg_hash_cache.csv`` is created inside ``src_dir``.
    hash_type, hash_size, hamming_threshold, use_ssim, ssim_threshold:
        Forwarded to :func:`_drop_duplicate_frames`.

    Returns
    -------
    int
        Number of files deleted as duplicates.
    """

    file_paths: list[str] = []
    for root, _, files in os.walk(src_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXT:
                file_paths.append(os.path.join(root, fname))

    if not file_paths:
        return 0

    df = pd.DataFrame({"SRC_PATH": file_paths})
    with ThreadPoolExecutor() as executor:
        sizes = list(executor.map(get_image_size, df["SRC_PATH"]))
    df["WIDTH"], df["HEIGHT"] = zip(*sizes)

    if cache_file is None:
        cache_file = os.path.join(src_dir, "sun_neg_hash_cache.csv")

    filtered_df = _drop_duplicate_frames(
        df,
        cache_file=cache_file,
        hash_type=hash_type,
        hash_size=hash_size,
        hamming_threshold=hamming_threshold,
        use_ssim=use_ssim,
        ssim_threshold=ssim_threshold,
    )

    keep: set[str] = set(filtered_df["SRC_PATH"])
    removed = 0
    for path in file_paths:
        if path not in keep:
            os.remove(path)
            removed += 1

    return removed


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


def _clip_bboxes(bboxes: pd.Series, widths: pd.Series, heights: pd.Series) -> pd.Series:
    """Vectorised clipping of bounding boxes to image dimensions.

    Args:
        bboxes: Series of space-separated ``x1,y1,x2,y2`` bounding boxes.
        widths: Series with corresponding image widths.
        heights: Series with corresponding image heights.

    Returns:
        Series with bounding boxes clamped to the image size.
    """

    result = []
    for bbox, w, h in zip(bboxes, widths, heights):
        if not bbox:
            result.append("")
            continue
        coords = np.fromstring(bbox.replace(" ", ","), sep=",", dtype=int).reshape(-1, 4)
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, w)
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, h)
        result.append(" ".join(",".join(map(str, c)) for c in coords))
    return pd.Series(result, index=bboxes.index)


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
    src_dir: str,
    dest_dir: str,
    metadata_filename: str = "metadata.csv",
    hash_type: str = "phash",
    hash_size: int = 16,
    hamming_threshold: int = 4,
    use_ssim: bool = False,
    ssim_threshold: float = 0.99,
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
        hash_type: Perceptual hash function from ``imagehash`` to use.
        hash_size: Size parameter for the perceptual hash function.
        hamming_threshold: Maximum Hamming distance to consider frames duplicates.
        use_ssim: Whether to confirm perceptual duplicates with SSIM.
        ssim_threshold: SSIM threshold when ``use_ssim`` is ``True``.

    Returns:
        None
    """

    logger.info("Converting SUN dataset from %s to %s", src_dir, dest_dir)
    metadata_path = os.path.join(src_dir, metadata_filename)
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.columns = [c.upper() for c in metadata_df.columns]
    metadata_df["SHAPE"] = metadata_df["SHAPE"].replace("IIa(LST-NG)", "IIa")
    metadata_df = metadata_df.drop(columns=["PATHOLOGICAL DIAGNOSIS"], errors="ignore")
    if "SIZE" in metadata_df.columns:
        metadata_df["SIZE"] = pd.to_numeric(
            metadata_df["SIZE"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce",
        )

    annotation_txt_dir = os.path.join(
        src_dir, "sundatabase_positive", "annotation_txt"
    )

    def _parse_case_annotation(args: tuple[str, pd.Series]) -> list[dict]:
        """Parse a single case annotation file into per-frame rows."""

        txt_file, meta = args
        case_rows: list[dict] = []
        with open(txt_file, "r") as f:
            for line in f:
                filename, bbox = _extract_bboxes(line)
                case_rows.append(
                    {
                        "FRAME": filename,
                        "POLYP ID": meta["ID"],
                        "LOCATION": meta.get("LOCATION"),
                        "SHAPE": meta.get("SHAPE"),
                        "SIZE": meta.get("SIZE"),
                        "BOUNDING BOX": bbox,
                    }
                )
        return case_rows

    annotation_args: list[tuple[str, pd.Series]] = []
    for _, row in metadata_df.iterrows():
        case_id = row["ID"]
        txt_file = os.path.join(annotation_txt_dir, f"case{case_id}.txt")
        if os.path.exists(txt_file):
            annotation_args.append((txt_file, row))

    with ThreadPoolExecutor() as executor:
        parsed_rows = executor.map(_parse_case_annotation, annotation_args)
    rows = [r for case_rows in parsed_rows for r in case_rows]

    df = pd.DataFrame(rows)

    df["SRC_PATH"] = df.apply(
        lambda r: os.path.join(
            src_dir, "sundatabase_positive", f"case{r['POLYP ID']}", r["FRAME"]
        ),
        axis=1,
    )

    with ThreadPoolExecutor() as executor:
        sizes = list(executor.map(get_image_size, df["SRC_PATH"]))
    df["WIDTH"], df["HEIGHT"] = zip(*sizes)

    cache_path = os.path.join(src_dir, "sun_hash_cache.csv")
    df = _drop_duplicate_frames(
        df,
        cache_file=cache_path,
        hash_type=hash_type,
        hash_size=hash_size,
        hamming_threshold=hamming_threshold,
        use_ssim=use_ssim,
        ssim_threshold=ssim_threshold,
    )
    df["BOUNDING BOX"] = _clip_bboxes(df["BOUNDING BOX"], df["WIDTH"], df["HEIGHT"])
    df["IMAGE TYPE"] = "WL"

    out_dir = os.path.join(dest_dir, "sun")
    img_dest_dir = os.path.join(out_dir, "img")
    move_images_parallel(
        None,
        img_dest_dir,
        df,
        filename_col="FRAME",
        source_path_col="SRC_PATH",
        copy=True,
    )

    df = df.drop(columns=["SRC_PATH"], errors="ignore")
    df.to_excel(os.path.join(out_dir, "annotations.xlsx"), index=False)
    validate_conversion(out_dir, df)
    logger.info("Finished SUN conversion with %d entries", len(df))
