import os
import shutil
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from .utils import draw_bounding_boxes, get_image_size
from tqdm import tqdm

from .preprocess.sun import _drop_duplicate_frames


logger = logging.getLogger(__name__)


# Supported image extensions for negative frame processing
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


# Mapping of dataset specific morphology descriptions to a common taxonomy
MORPHOLOGY_SHAPE_MAP: dict[str, str] = {
    "is": "sessile",
    "sessile": "sessile",
    "iia": "flat",
    "iib": "flat",
    "flat": "flat",
    "flat elevated": "flat",
    "ip": "pedunculated",
    "pedunculated": "pedunculated",
    "iic": "depressed",
    "depressed": "depressed",
}


def generate_global_lesion_ids(df: pd.DataFrame) -> list[str]:
    """Generate collision-proof lesion identifiers across datasets.

    The resulting identifiers keep the original lesion numbers while
    prefixing them with the dataset name to avoid clashes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``ORIGINAL DATASET`` and ``POLYP ID`` columns.

    Returns
    -------
    list[str]
        Identifiers in ``<DATASET>_<LESION_ID>`` format. ``LESION_ID`` is
        zero-padded per dataset to preserve a consistent length.
    """

    pad_width = {"SUN": 4, "POLAR": 5, "PICCOLO": 5}
    new_ids: list[str] = []

    for _, row in df.iterrows():
        dataset = str(row["ORIGINAL DATASET"]).upper()
        lesion_id = str(row["POLYP ID"])

        if lesion_id.isdigit():
            lesion_id = lesion_id.zfill(pad_width.get(dataset, 5))

        new_ids.append(f"{dataset}_{lesion_id}")

    return new_ids

def combine_datasets(base_path, datasets, num_elements=None, random_state=None):
    """Merge datasets into a single DataFrame, optionally including a specified
    number of elements from each dataset.

    Args:
    - base_path (str): The base path where the dataset folders are located.
    - datasets (list of str): A list of dataset names to merge.
    - num_elements (int, optional): The number of elements to include from each dataset.
      If ``None`` (default), all rows are included.
    - random_state (int, optional): Seed for the random number generator used when
      sampling rows.

    Returns:
    - pd.DataFrame: The combined DataFrame with an 'ORIGINAL DATASET' column indicating the source dataset
      for each row. If ``num_elements`` is specified and is less than the number of available rows
      in a dataset, rows are randomly selected to meet the specified count.
    """
    dataframes = []

    for dataset in datasets:
        # Datasets converted to the standard layout may either reside directly
        # under ``base_path`` or within a ``raw`` sub-directory.  Prefer the
        # latter for backwards compatibility but fall back to the former when
        # needed.
        candidates = [
            os.path.join(base_path, "raw", dataset, "annotations.xlsx"),
            os.path.join(base_path, dataset, "annotations.xlsx"),
        ]
        spreadsheet_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        try:
            df = pd.read_excel(spreadsheet_path)
            original_len = len(df)
            df["ORIGINAL DATASET"] = dataset

            logger.info(
                "Loaded %d rows from %s", original_len, spreadsheet_path
            )

            # Randomly select 'num_elements' rows if specified and dataset is larger
            if num_elements is not None and original_len > num_elements:
                df = df.sample(n=num_elements, random_state=random_state)
                logger.info(
                    "Dataset %s truncated to %d rows (dropped %d)",
                    dataset,
                    num_elements,
                    original_len - num_elements,
                )

            dataframes.append(df)
        except FileNotFoundError:
            logger.warning("Spreadsheet not found: %s", spreadsheet_path)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def save_combined_dataframe(combined_df, base_path, filename='annotations.xlsx'):
    output_path = os.path.join(base_path, filename)
    combined_df.to_excel(output_path, index=False)
    logger.info("The combined DataFrame has been saved to %s", output_path)


def harmonize_metadata(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """Convert raw merged annotations to the unified schema.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`merge_datasets` prior to harmonisation.
        It must contain ``global_lesion_id``, ``lesion_id``, ``FRAME`` and
        ``ORIGINAL FRAME`` columns.
    image_dir:
        Directory where the copied images reside.  File paths are constructed by
        joining this directory with the frame filenames.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns following the unified schema described in
        :mod:`polypdb.schema`.
    """

    df = df.copy()

    df['dataset'] = df['ORIGINAL DATASET'].str.lower()
    df['frame'] = df['FRAME']
    df['original_frame'] = df.get('ORIGINAL FRAME', df['FRAME'])
    df['n_frames'] = df.groupby('global_lesion_id')['frame'].transform('count').astype(int)
    df['image_paths'] = [
        [os.path.join(image_dir, fname)] for fname in df['frame']
    ]

    if 'BOUNDING BOX' in df.columns:
        def _parse_bbox(val: str) -> list[list[int]]:
            if pd.isna(val):
                return []
            boxes = []
            for part in str(val).split(';'):
                coords = [int(float(c)) for c in part.split(',') if c]
                if len(coords) == 4:
                    boxes.append(coords)
            return boxes

        df['bounding_boxes'] = df['BOUNDING BOX'].apply(_parse_bbox)
    else:
        df['bounding_boxes'] = [[] for _ in range(len(df))]

    if 'SHAPE' in df.columns:
        # Preserve original morphology labels verbatim (e.g., 'Is', 'Ip', 'IIa')
        df['morphology_shape'] = df['SHAPE']
    else:
        df['morphology_shape'] = None

    df['morphology_size'] = df.get('SIZE')

    if 'IMAGE TYPE' in df.columns:
        df['image_type'] = df['IMAGE TYPE'].str.lower()
    else:
        df['image_type'] = None

    df['morphology_location'] = df.get('LOCATION')
    df['center_id'] = df.get('CENTER')

    # Drop legacy columns now represented in the unified schema
    drop_cols = [
        'ORIGINAL DATASET',
        'FRAME',
        'ORIGINAL FRAME',
        'SHAPE',
        'SIZE',
        'LOCATION',
        'IMAGE TYPE',
        'BOUNDING BOX',
        'POLYP ID',  # global id used for naming during processing
        'CENTER',
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    core_cols = [
        'global_lesion_id',
        'dataset',
        'lesion_id',
        'n_frames',
        'frame',
        'original_frame',
        'image_paths',
        'bounding_boxes',
        'morphology_shape',
        'morphology_size',
        'image_type',
        'morphology_location',
        'center_id',
    ]

    other_cols = [c for c in df.columns if c not in core_cols]
    return df[core_cols + other_cols]


def validate_harmonised_dataframe(df: pd.DataFrame, image_dir: str) -> None:
    """Validate integrity of a harmonised dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame produced by :func:`harmonize_metadata`.
    image_dir : str
        Directory containing the harmonised image files.

    Raises
    ------
    ValueError
        If required columns are missing, contain null values or duplicate
        ``global_lesion_id`` values are detected.
    FileNotFoundError
        If any path in ``image_paths`` does not exist inside ``image_dir``.
    """

    required_columns = [
        "global_lesion_id",
        "frame",
        "original_frame",
        "image_paths",
        "bounding_boxes",
        "dataset",
        "lesion_id",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Null values found in required columns")

    abs_image_dir = os.path.abspath(image_dir)
    for paths in df["image_paths"]:
        for path in paths:
            abs_path = os.path.abspath(path)
            if os.path.commonpath([abs_path, abs_image_dir]) != abs_image_dir:
                raise ValueError(f"Image path outside destination directory: {path}")
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Referenced image not found: {path}")

    for bboxes in df["bounding_boxes"]:
        for box in bboxes:
            if not (
                isinstance(box, (list, tuple))
                and len(box) == 4
                and all(isinstance(c, int) for c in box)
            ):
                raise ValueError(f"Invalid bounding box format: {box}")


def copy_and_rename_images(df, base_path, dest_dir):
    """Copy images from source datasets and assign deterministic names.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``ORIGINAL DATASET``, ``POLYP ID`` and ``FRAME`` columns.
    base_path : str
        Root directory containing the dataset folders.
    dest_dir : str
        Destination directory where renamed images will be stored.
    """
    os.makedirs(dest_dir, exist_ok=True)  # Ensure the destination directory exists
    counters: defaultdict[str, int] = defaultdict(int)
    missing_files = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        dataset_name = str(row["ORIGINAL DATASET"])
        dataset = dataset_name.lower()

        # Look for the preprocessed images either in ``base_path/raw`` or
        # directly under ``base_path`` to support different directory layouts.
        candidates = [
            os.path.join(base_path, "raw", dataset_name, "img", row["FRAME"]),
            os.path.join(base_path, dataset_name, "img", row["FRAME"]),
        ]
        source_image_path = next((p for p in candidates if os.path.exists(p)), None)

        if source_image_path is not None:
            global_id = row["POLYP ID"]
            counters[global_id] += 1
            frame_idx = counters[global_id]
            ext = os.path.splitext(row["FRAME"])[1].lower()
            global_frame_id = f"{global_id}.{frame_idx:06d}"
            new_filename = f"{global_frame_id}{ext}"
            dest_file_path = os.path.join(dest_dir, new_filename)
            shutil.copy(source_image_path, dest_file_path)
            df.at[index, "ORIGINAL FRAME"] = row["FRAME"]
            df.at[index, "FRAME"] = new_filename
            df.at[index, "global_frame_id"] = global_frame_id
            df.at[index, "raw_path"] = source_image_path
            df.at[index, "dataset"] = dataset
        else:
            missing_files += 1
            logger.warning("File not found: %s", candidates[0])

    if missing_files:
        logger.info("Skipped %d images due to missing files", missing_files)


def copy_and_rename_negative_images(
    source_dir: str, dest_dir: str, dataset_name: str | None = None
) -> tuple[list[dict], int]:
    """Copy negative frames into the canonical image directory.

    Negative sources may either contain subdirectories that represent
    deterministic *groups* (e.g., video IDs) or store images directly under the
    dataset directory. When images live at the root, the directory name itself
    is used as the group label.

    Two directory layouts are therefore supported:

    - ``source_dir/<dataset>/<group>`` where ``<group>`` is a deterministic
      identifier such as a clip or acquisition date. The dataset name is read
      from the directory structure.
    - ``source_dir/<group>`` while explicitly providing ``dataset_name``. Images
      may be directly inside ``source_dir``; in that case ``<group>`` defaults to
      the basename of ``source_dir``.

    Each image is renamed using ``NEG_<DATASET>_<GROUP>.<frame_idx>.<ext>`` and
    copied to ``dest_dir``. Metadata records for the copied frames are returned
    to be included in ``frames.parquet``.

    Parameters
    ----------
    source_dir:
        Root directory containing negative images organised by dataset and
        optionally by group, or only by group when ``dataset_name`` is given.
    dest_dir:
        Canonical images directory.
    dataset_name:
        Optional dataset identifier when ``source_dir`` already points to a
        dataset‑specific directory.

    Returns
    -------
    tuple[list[dict], int]
        Tuple of (records, original_count) where ``records`` describes the
        copied frames and ``original_count`` indicates how many candidate images
        were seen before dropping duplicates.
    """

    records: list[dict] = []
    if not os.path.isdir(source_dir):
        return records, 0

    os.makedirs(dest_dir, exist_ok=True)

    if dataset_name is not None:
        dataset_entries = [(dataset_name, source_dir)]
    else:
        dataset_entries = [
            (name, os.path.join(source_dir, name))
            for name in sorted(os.listdir(source_dir))
            if os.path.isdir(os.path.join(source_dir, name))
        ]

    # gather candidate file paths
    candidates: list[dict] = []
    for ds_name, dataset_path in dataset_entries:
        entries = sorted(os.listdir(dataset_path))
        subdirs = [e for e in entries if os.path.isdir(os.path.join(dataset_path, e))]
        root_files = [
            e for e in entries
            if os.path.isfile(os.path.join(dataset_path, e))
            and os.path.splitext(e)[1].lower() in IMG_EXT
        ]
        groups: list[str | None] = list(subdirs)
        if root_files:
            groups.append(None)

        for group in groups:
            if group is None:
                group_path = dataset_path
                group_name = os.path.basename(dataset_path)
                files = root_files
            else:
                group_path = os.path.join(dataset_path, group)
                group_name = group
                files = [
                    f for f in sorted(os.listdir(group_path))
                    if os.path.isfile(os.path.join(group_path, f))
                    and os.path.splitext(f)[1].lower() in IMG_EXT
                ]

            for fname in files:
                candidates.append(
                    {
                        "SRC_PATH": os.path.join(group_path, fname),
                        "dataset": ds_name.lower(),
                        "group": group_name,
                        "filename": fname,
                    }
                )

    if not candidates:
        return records, 0

    df = pd.DataFrame(candidates)
    with ThreadPoolExecutor() as executor:
        sizes = list(executor.map(get_image_size, df["SRC_PATH"]))
    df["WIDTH"], df["HEIGHT"] = zip(*sizes)

    original_count = len(df)
    logger.info(
        "Found %d negative images before dropping duplicates", original_count
    )

    cache_file = os.path.join(source_dir, "neg_hash_cache.csv")
    filtered_df = _drop_duplicate_frames(df, cache_file=cache_file)
    filtered_df = filtered_df.sort_values(["dataset", "group", "filename"])
    logger.info(
        "Retained %d negative images after dropping duplicates", len(filtered_df)
    )

    for (ds_name, group_name), group_df in filtered_df.groupby(["dataset", "group"]):
        lesion_key = f"NEG_{ds_name.upper()}_{group_name}"
        frame_idx = 0
        for _, row in group_df.iterrows():
            frame_idx += 1
            src = row["SRC_PATH"]
            fname = row["filename"]
            ext = os.path.splitext(fname)[1].lower()
            global_frame_id = f"{lesion_key}.{frame_idx:06d}"
            dest_name = f"{global_frame_id}{ext}"
            dest_path = os.path.join(dest_dir, dest_name)
            shutil.copy(src, dest_path)
            records.append(
                {
                    "global_frame_id": global_frame_id,
                    # Group negatives by synthetic case key
                    "global_case_id": lesion_key,
                    "dataset": ds_name,
                    "source_frame_id": fname,
                    "img_width": row["WIDTH"],
                    "img_height": row["HEIGHT"],
                    "filename_ext": ext,
                    "img_path": os.path.join('images', dest_name),
                    "modality": None,
                }
            )

    return records, original_count


def calculate_target_size(df, width_percentile=0, height_percentile=0):
    """
    Calculate target width and height for image standardization based on percentiles.

    :param df: DataFrame containing image metadata.
    :param width_percentile: Percentile to use for width calculation.
    :param height_percentile: Percentile to use for height calculation.
    :return: target width and height as integers.
    :rtype: tuple of integers (target_width, target_height)
    """
    # Calculate aspect ratios
    aspect_ratios = df['WIDTH'] / df['HEIGHT']

    # Calculate median aspect ratio
    median_aspect_ratio = np.median(aspect_ratios)

    # Calculate width and height based on the specified percentiles
    target_width = int(np.percentile(df['WIDTH'], width_percentile))
    target_height = int(np.percentile(df['HEIGHT'], height_percentile))

    # Adjust target dimensions to maintain the median aspect ratio
    if median_aspect_ratio > 1:
        # Landscape orientation, adjust width based on aspect ratio and height percentile
        target_width = max(target_width, int(target_height * median_aspect_ratio))
    else:
        # Portrait orientation, adjust height based on aspect ratio and width percentile
        target_height = max(target_height, int(target_width / median_aspect_ratio))

    return target_width, target_height


def adjust_bounding_box(original_bbox, original_width, original_height, target_width, target_height, result_width,
                        result_height):
    """
    Adjusts bounding box coordinates based on image standardization.
    :param original_bbox: Original bounding box coordinates as a string
    :param original_width: Original image width
    :param original_height: Original image height
    :param target_width: Target image width
    :param target_height: Target image height
    :param result_width: Resulting image width after standardization
    :param result_height: Resulting image height after standardization
    :return: Adjusted bounding box coordinates as a string
    """
    x_min, y_min, x_max, y_max = map(int, original_bbox.split(','))

    # Calculate scale factors
    scale_x = result_width / original_width
    scale_y = result_height / original_height

    # Calculate padding added
    pad_x = (target_width - result_width) // 2
    pad_y = (target_height - result_height) // 2

    # Scale and adjust for padding
    x_min = int(x_min * scale_x + pad_x)
    x_max = int(x_max * scale_x + pad_x)
    y_min = int(y_min * scale_y + pad_y)
    y_max = int(y_max * scale_y + pad_y)

    return f"{x_min},{y_min},{x_max},{y_max}"


def standardize_and_override_image(img_path, target_width, target_height):
    """
    Standardizes the image and overrides it with a new image with the specified width and height.
    :param img_path: Path to the input image.
    :param target_width: Desired width of the standardized image.
    :param target_height: Desired height of the standardized image.
    :return: Tuple containing original width, original height, new width, and new height.
    """
    with Image.open(img_path) as img:
        original_width, original_height = img.size
        img_aspect = original_width / original_height
        target_aspect = target_width / target_height

        if img_aspect > target_aspect:
            # Wider than target aspect
            new_height = target_height
            new_width = int(new_height * img_aspect)
        else:
            # Taller or equal to target aspect
            new_width = target_width
            new_height = int(new_width / img_aspect)

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Add padding if necessary
        result_img = Image.new("RGB", (target_width, target_height), "black")
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result_img.paste(resized_img, (paste_x, paste_y))

        result_img.save(img_path, quality=95)

        return original_width, original_height, new_width, new_height


def process_images_and_bboxes(df, source_dir, target_width, target_height):
    """
    Standardizes and adjusts bounding box coordinates for images in the DataFrame.
    :param df: DataFrame containing image metadata and bounding box coordinates.
    :param source_dir: Directory containing the original images.
    :param target_width: Desired width of the standardized images.
    :param target_height: Desired height of the standardized images.
    """
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Standardizing images"):
        img_path = os.path.join(source_dir, row['FRAME'])
        if os.path.exists(img_path):
            orig_width, orig_height, result_width, result_height = standardize_and_override_image(img_path,
                                                                                                  target_width,
                                                                                                  target_height)
            new_bbox = None
            bbox_val = str(row['BOUNDING BOX']) if not pd.isna(row['BOUNDING BOX']) else ''
            if bbox_val:
                parts = re.split(r"[;\s]+", bbox_val.strip())
                new_parts: list[str] = []
                for p in parts:
                    if not p:
                        continue
                    new_parts.append(
                        adjust_bounding_box(
                            p,
                            orig_width,
                            orig_height,
                            target_width,
                            target_height,
                            result_width,
                            result_height,
                        )
                    )
                new_bbox = ";".join(new_parts)
            else:
                new_bbox = ''
            df.at[index, 'BOUNDING BOX'] = new_bbox
        else:
            logger.warning("File not found: %s", img_path)


def process_negative_images(source_dir, target_width, target_height):
    """Standardize images in ``source_dir`` without bounding boxes.

    The function walks the directory tree recursively so negatives may be
    organised in arbitrary subfolders.

    Parameters
    ----------
    source_dir : str
        Directory containing images to be resized.
    target_width : int
        Desired output width.
    target_height : int
        Desired output height.
    """
    IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    if os.path.exists(source_dir):
        file_paths: list[str] = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IMG_EXT:
                    file_paths.append(os.path.join(root, file))
        for source_file_path in tqdm(file_paths, desc="Processing negatives", total=len(file_paths)):
            standardize_and_override_image(source_file_path, target_width, target_height)


def cleanup_temp_artifacts(base_path: str, destination_folder: str) -> None:
    """Remove temporary directories created during dataset merging.

    Parameters
    ----------
    base_path : str
        Root directory for dataset processing.
    destination_folder : str
        Folder where processed images were stored. Temporary artifacts
        may exist alongside or within this directory.
    """

    temp_dirs = [
        os.path.join(base_path, 'img_bbox'),
        os.path.join(base_path, destination_folder, 'resized'),
        os.path.join(base_path, destination_folder, 'bbox_preview'),
    ]

    for temp_dir in temp_dirs:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Removed temporary directory %s", temp_dir)


def export_parquet_metadata(
    df: pd.DataFrame, canonical_base: str, negative_frames: list[dict] | None = None
) -> None:
    """Write canonical metadata tables to Parquet files.

    Parameters
    ----------
    df : pandas.DataFrame
        Harmonised metadata produced by :func:`harmonize_metadata`.
    canonical_base : str
        Base directory for the canonical dataset. ``images`` and ``manifests``
        sub-directories are expected beneath this path.
    """

    images_dir = os.path.join(canonical_base, 'images')
    manifests_dir = os.path.join(canonical_base, 'manifests')
    os.makedirs(manifests_dir, exist_ok=True)

    # Note: ingest metadata removed per user request

    frame_records = []
    for _, row in df.iterrows():
        canonical_path = os.path.join(images_dir, row['frame'])
        with Image.open(canonical_path) as img:
            width, height = img.size
        frame_records.append({
            'global_frame_id': row.get('global_frame_id'),
            'global_case_id': row['global_lesion_id'],
            'dataset': row['dataset'],
            'center_id': row.get('center_id'),
            'source_frame_id': row['original_frame'],
            'img_width': width,
            'img_height': height,
            'filename_ext': os.path.splitext(row['frame'])[1].lower(),
            'img_path': os.path.join('images', row['frame']),
            'modality': row.get('image_type'),
            'has_polyp': True,
        })

    if negative_frames:
        for rec in negative_frames:
            rec = rec.copy()
            rec.update({'has_polyp': False})
            frame_records.append(rec)
    pd.DataFrame(frame_records).to_parquet(
        os.path.join(manifests_dir, 'frames.parquet'), index=False
    )

    case_records = []
    for lesion_id, group in df.groupby('global_lesion_id'):
        first = group.iloc[0]
        case_records.append({
            'global_case_id': lesion_id,
            'dataset': first['dataset'],
            'center_id': first.get('center_id'),
            'shape': first.get('morphology_shape'),
            'size_mm': first.get('morphology_size'),
            'location': first.get('morphology_location'),
            'n_frames': int(first['n_frames']),
            'source_lesion_id': first['lesion_id'],
        })
    pd.DataFrame(case_records).to_parquet(
        os.path.join(manifests_dir, 'cases.parquet'), index=False
    )

    ann_records = []
    for _, row in df.iterrows():
        for idx, bbox in enumerate(row['bounding_boxes']):
            x1, y1, x2, y2 = bbox
            ann_records.append({
                'ann_id': f"{row.get('global_frame_id')}.{idx:02d}",
                'global_frame_id': row.get('global_frame_id'),
                'global_case_id': row['global_lesion_id'],
                'center_id': row.get('center_id'),
                'bbox_x': float(x1),
                'bbox_y': float(y1),
                'bbox_w': float(x2 - x1),
                'bbox_h': float(y2 - y1),
                'bbox_format': 'xywh',
                'bbox_normalized': False,
            })
    pd.DataFrame(ann_records).to_parquet(
        os.path.join(manifests_dir, 'annotations.parquet'), index=False
    )


def dataset_statistics(
    df: pd.DataFrame, negative_frames: list[dict], negative_original_count: int
) -> dict:
    """Compute and log summary statistics for the fused dataset.

    Parameters
    ----------
    df:
        DataFrame containing positive frame metadata.
    negative_frames:
        List of dictionaries describing negative frames. Each record must
        include ``global_frame_id`` and ``dataset`` keys.
    negative_original_count:
        Number of negative frames before duplicate removal.

    Returns
    -------
    dict
        A dictionary with overall dataset metrics. In addition to the
        previous summary values, the result now includes
        ``dataset_case_distribution`` and ``dataset_frame_distribution``
        showing positive/negative counts per source dataset.
    """
    total_positive = len(df)
    total_negative = len(negative_frames)
    original_total = total_positive + negative_original_count
    total_after = total_positive + total_negative

    positive_cases = df["global_case_id"].nunique()
    neg_case_ids = [rec["global_frame_id"].split(".")[0] for rec in negative_frames]
    negative_cases = len(set(neg_case_ids))
    total_cases = positive_cases + negative_cases

    case_distribution = {"positive": positive_cases, "negative": negative_cases}

    morph_counts = (
        df.drop_duplicates("global_case_id")["morphology_shape"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )

    # Positive and negative case counts per dataset
    pos_case_by_dataset = df.drop_duplicates("global_case_id")["dataset"].value_counts().to_dict()
    from collections import Counter
    neg_case_map: dict[str, str] = {}
    for rec in negative_frames:
        cid = rec["global_frame_id"].split(".")[0]
        neg_case_map[cid] = rec["dataset"]
    neg_case_by_dataset = Counter(neg_case_map.values())
    dataset_counts = Counter(pos_case_by_dataset)
    dataset_counts.update(neg_case_by_dataset)
    dataset_counts = dict(dataset_counts)

    dataset_case_distribution = {}
    for ds in set(pos_case_by_dataset) | set(neg_case_by_dataset):
        dataset_case_distribution[ds] = {
            "positive": int(pos_case_by_dataset.get(ds, 0)),
            "negative": int(neg_case_by_dataset.get(ds, 0)),
        }

    pos_frame_counts = df.groupby("global_case_id").size()
    neg_frame_counts = Counter(neg_case_ids)
    all_frame_counts = pd.concat([pos_frame_counts, pd.Series(neg_frame_counts)])

    pos_frames_by_dataset = df["dataset"].value_counts().to_dict()
    neg_frames_by_dataset = Counter(rec["dataset"] for rec in negative_frames)
    dataset_frame_distribution = {}
    for ds in set(pos_frames_by_dataset) | set(neg_frames_by_dataset):
        dataset_frame_distribution[ds] = {
            "positive": int(pos_frames_by_dataset.get(ds, 0)),
            "negative": int(neg_frames_by_dataset.get(ds, 0)),
        }
    frame_summary = {
        "min": int(all_frame_counts.min()),
        "max": int(all_frame_counts.max()),
        "mean": float(all_frame_counts.mean()),
        "median": float(all_frame_counts.median()),
    }

    stats = {
        "original_total_samples": original_total,
        "total_samples": total_after,
        "total_cases": total_cases,
        "case_distribution": case_distribution,
        "morphology_distribution": morph_counts,
        "dataset_distribution": dataset_counts,
        "frames_per_case_summary": frame_summary,
        "dataset_case_distribution": dataset_case_distribution,
        "dataset_frame_distribution": dataset_frame_distribution,
    }

    logger.info("Original total samples: %d", original_total)
    logger.info("Total samples after dropping duplicates: %d", total_after)
    logger.info("Total cases: %d", total_cases)
    logger.info("Case distribution: %s", case_distribution)
    logger.info("Morphology distribution: %s", morph_counts)
    logger.info("Dataset distribution: %s", dataset_counts)
    logger.info("Dataset case distribution: %s", dataset_case_distribution)
    logger.info("Dataset frame distribution: %s", dataset_frame_distribution)
    logger.info("Frames per case summary: %s", frame_summary)

    return stats


def dataset_statistics_from_canonical(canonical_base: str) -> dict:
    """Compute dataset statistics from a canonical dataset directory.

    Parameters
    ----------
    canonical_base : str
        Path to the canonical dataset containing a ``manifests`` directory
        with ``frames.parquet`` and ``cases.parquet`` files.

    Returns
    -------
    dict
        Summary statistics in the same format as :func:`dataset_statistics`.
    """

    manifests_dir = os.path.join(canonical_base, "manifests")
    frames_path = os.path.join(manifests_dir, "frames.parquet")
    cases_path = os.path.join(manifests_dir, "cases.parquet")

    frames_df = pd.read_parquet(frames_path)
    cases_df = pd.read_parquet(cases_path)

    pos_df = frames_df[frames_df["has_polyp"]].merge(
        cases_df[["global_case_id", "dataset", "shape"]],
        on=["global_case_id", "dataset"],
        how="left",
    )
    pos_df = pos_df.rename(columns={"shape": "morphology_shape"})

    neg_df = frames_df[~frames_df["has_polyp"]]
    negative_records = neg_df[["global_frame_id", "dataset"]].to_dict("records")

    return dataset_statistics(pos_df, negative_records, negative_original_count=len(neg_df))


def dataset_statistics_from_parquets(frames_path: str, cases_path: str) -> dict:
    """Compute dataset statistics given explicit manifest parquet paths.

    Parameters
    ----------
    frames_path:
        Path to ``frames.parquet`` containing per-frame metadata with a
        boolean ``has_polyp`` column.
    cases_path:
        Path to ``cases.parquet`` containing per-case metadata including
        ``global_case_id``, ``dataset`` and morphology fields.

    Returns
    -------
    dict
        Summary statistics in the same format as :func:`dataset_statistics`.
    """

    frames_df = pd.read_parquet(frames_path)
    cases_df = pd.read_parquet(cases_path)

    pos_df = frames_df[frames_df["has_polyp"]].merge(
        cases_df[["global_case_id", "dataset", "shape"]],
        on=["global_case_id", "dataset"],
        how="left",
    )
    pos_df = pos_df.rename(columns={"shape": "morphology_shape"})

    neg_df = frames_df[~frames_df["has_polyp"]]
    negative_records = neg_df[["global_frame_id", "dataset"]].to_dict("records")

    return dataset_statistics(pos_df, negative_records, negative_original_count=len(neg_df))


def export_manifests_to_csv(
    canonical_base: str, out_dir: str, tables: list[str] | None = None
) -> dict[str, str]:
    """Export manifest parquet files to CSV.

    Parameters
    ----------
    canonical_base:
        Path to the canonical dataset directory that contains a ``manifests``
        subdirectory with parquet files.
    out_dir:
        Destination directory where CSV files will be written.
    tables:
        Optional list of tables to export. Supported values: ``frames``,
        ``lesions``, ``annotations``. If ``None``, all available tables are
        exported.

    Returns
    -------
    dict
        Mapping from table name to the written CSV path.
    """

    manifests_dir = os.path.join(canonical_base, "manifests")
    os.makedirs(out_dir, exist_ok=True)

    supported = ["frames", "cases", "annotations"]
    if tables is None:
        tables = supported
    else:
        invalid = [t for t in tables if t not in supported]
        if invalid:
            raise ValueError(f"Unsupported tables requested: {invalid}")

    written: dict[str, str] = {}
    for t in tables:
        parquet_path = os.path.join(manifests_dir, f"{t}.parquet")
        if not os.path.exists(parquet_path):
            logger.warning("Parquet not found for table %s: %s (skipping)", t, parquet_path)
            continue
        df = pd.read_parquet(parquet_path)
        csv_path = os.path.join(out_dir, f"{t}.csv")
        df.to_csv(csv_path, index=False)
        written[t] = csv_path
        logger.info("Wrote %s CSV to %s", t, csv_path)

    if not written:
        logger.warning("No CSV files were written; check your canonical directory")
    return written


def merge_datasets(
    base_path,
    datasets: list[str],
    num_elements=None,
    plot_images=False,
    negative_sources: dict[str, str] | None = None,
):
    """Merge and standardize multiple datasets.

    Parameters
    ----------
    base_path : str
        Base directory where the dataset folders are located.
    datasets : list of str
        Names of datasets to merge.
    num_elements : int, optional
        Number of samples to include from each dataset. All samples are used if ``None``.
    plot_images : bool, optional
        If ``True``, bounding boxes are drawn on images for visual inspection.
    negative_sources : dict, optional
        Mapping of dataset names to directories containing negative frames. Paths
        may be absolute or relative to ``base_path`` and must contain
        sub‑directories for deterministic groups (e.g. video ids).
    """
    # process_negative_images(base_path, 588, 478)

    combined_df = combine_datasets(base_path, datasets, num_elements=num_elements)
    output_filename = 'metadata.xlsx' if num_elements is None else 'output_' + str(num_elements) + '.xlsx'

    # Keep only entries with a single bounding box for non-PolypGen datasets.
    # PolypGen can legitimately contain multiple boxes per frame; preserve them.
    before_bb_filter = len(combined_df)
    is_polypgen = combined_df['ORIGINAL DATASET'].str.lower() == 'polypgen'
    has_multiple = combined_df['BOUNDING BOX'].astype(str).str.contains(r"[;\s]")
    combined_df = combined_df[is_polypgen | ~has_multiple]
    dropped_bb = before_bb_filter - len(combined_df)
    if dropped_bb:
        logger.info(
            "Dropped %d rows with multiple bounding boxes (non-PolypGen)", dropped_bb
        )

    logger.info("Combined dataframe length: %d", len(combined_df))

    # Preserve original lesion identifiers and create global ids used across datasets
    combined_df['lesion_id'] = combined_df['POLYP ID'].astype(str)
    combined_df['global_lesion_id'] = generate_global_lesion_ids(combined_df)
    combined_df['POLYP ID'] = combined_df['global_lesion_id']

    # Copy and rename images into canonical layout
    logger.info("COPYING IMAGES")
    canonical_base = os.path.join(base_path, 'canonical')
    images_dir = os.path.join(canonical_base, 'images')
    copy_and_rename_images(combined_df, base_path, images_dir)

    # Calculate target size
    logger.info("CALCULATING TARGET SIZE")
    target_width, target_height = calculate_target_size(combined_df)
    logger.info("Target size: %s x %s", target_width, target_height)

    # Standardize and adjust images and bounding boxes
    logger.info("STANDARDIZING IMAGES")
    process_images_and_bboxes(combined_df, images_dir, target_width, target_height)

    negative_records: list[dict] = []
    negative_original_count = 0
    if num_elements is None and negative_sources:
        for ds_name, src in negative_sources.items():
            src_dir = os.path.expanduser(src)
            if not os.path.isabs(src_dir):
                src_dir = os.path.join(base_path, src_dir)
            process_negative_images(src_dir, target_width, target_height)
            recs, orig_count = copy_and_rename_negative_images(
                src_dir, images_dir, dataset_name=ds_name
            )
            negative_records.extend(recs)
            negative_original_count += orig_count

    combined_df.drop(columns=['WIDTH', 'HEIGHT'], inplace=True)

    # Harmonise column names and add derived metadata
    combined_df = harmonize_metadata(combined_df, images_dir)

    validate_harmonised_dataframe(combined_df, images_dir)

    # Save the combined DataFrame to a spreadsheet and Parquet manifests
    save_combined_dataframe(combined_df, base_path, output_filename)
    export_parquet_metadata(combined_df, canonical_base, negative_records)

    # Compute statistics using the new 'global_case_id' terminology without fallbacks
    stats_df = combined_df.rename(columns={"global_lesion_id": "global_case_id"})
    dataset_statistics(stats_df, negative_records, negative_original_count)

    # Loop through the DataFrame and draw bounding boxes on each image
    if plot_images:
        # Create the destination directory if it doesn't exist
        bbox_dest_dir = os.path.join(base_path, 'img_bbox')
        os.makedirs(bbox_dest_dir, exist_ok=True)
        for index, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Drawing bounding boxes"):
            image_filename = row['frame']
            input_frame_path = os.path.join(images_dir, image_filename)
            output_frame_path = os.path.join(bbox_dest_dir, image_filename)
            draw_bounding_boxes(row['BOUNDING BOX'], input_frame_path, output_frame_path)

    cleanup_temp_artifacts(base_path, os.path.join('canonical', 'images'))
