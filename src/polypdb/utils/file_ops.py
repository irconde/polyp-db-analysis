"""File and data movement helpers.

Utilities in this module assist with relocating image files and validating
conversion results.  They were previously part of :mod:`polypdb.utils`.
"""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


logger = logging.getLogger(__name__)


def move_images(source_dir, dest_dir, df, filename_col="FRAME"):
    """Move images from ``source_dir`` to ``dest_dir`` based on ``df``."""

    os.makedirs(dest_dir, exist_ok=True)

    for _, row in df.iterrows():
        frame_filename = row[filename_col]
        source_path = os.path.join(source_dir, frame_filename)
        destination_path = os.path.join(dest_dir, frame_filename)

        try:
            shutil.move(source_path, destination_path)
        except IOError as e:  # pragma: no cover - logging only
            logger.error("Could not move %s. Error: %s", frame_filename, e)


def move_images_parallel(
    source_dir,
    dest_dir,
    df,
    filename_col="FRAME",
    source_path_col=None,
    copy=False,
):
    """Move images concurrently using :class:`ThreadPoolExecutor`."""

    os.makedirs(dest_dir, exist_ok=True)

    cols = [filename_col]
    if source_path_col:
        cols.append(source_path_col)

    rows = df[cols].itertuples(index=False, name=None)

    total = len(df)
    action = "copy" if copy else "move"
    logger.info("%s %d images to %s", action.capitalize(), total, dest_dir)

    def _transfer(row):
        if source_path_col:
            frame_filename, source_path = row
        else:
            (frame_filename,) = row
            source_path = os.path.join(source_dir, frame_filename)
        destination_path = os.path.join(dest_dir, frame_filename)
        try:
            if copy:
                shutil.copy(source_path, destination_path)
            else:
                shutil.move(source_path, destination_path)
        except IOError as e:  # pragma: no cover - logging only
            logger.error("Could not %s %s. Error: %s", action, frame_filename, e)

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(_transfer, rows),
                total=total,
                desc=f"{action.title()}ing images",
            )
        )
    logger.info("Finished %s images", action)


def validate_conversion(dest_dir, df, filename_col="FRAME"):
    """Validate that moved images match entries in ``df``."""

    img_dir = os.path.join(dest_dir, "img")
    try:
        files = [
            f
            for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
        ]
    except FileNotFoundError:
        logger.error("Image directory %s does not exist", img_dir)
        raise

    df_count = len(df)
    file_count = len(files)

    duplicates = df[df[filename_col].duplicated()][filename_col].tolist()
    if duplicates:
        logger.error(
            "Duplicated filenames in metadata: %s", ", ".join(sorted(duplicates))
        )

    if df_count != file_count:
        df_filenames = set(df[filename_col].astype(str))
        file_set = set(files)
        missing = sorted(df_filenames - file_set)
        extra = sorted(file_set - df_filenames)
        if missing:
            logger.error("Missing images: %s", ", ".join(missing))
        if extra:
            logger.error("Unexpected images: %s", ", ".join(extra))
        error_msg = (
            f"{df_count} rows in annotations but found {file_count} images in {img_dir}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        "Validated conversion for %s: %d rows match %d images", dest_dir, df_count, file_count
    )


__all__ = [
    "move_images",
    "move_images_parallel",
    "validate_conversion",
]

