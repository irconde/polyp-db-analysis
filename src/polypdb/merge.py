import os
import shutil
import logging

import numpy as np
import pandas as pd
from PIL import Image

from .utils import draw_bounding_boxes
import uuid
from tqdm import tqdm


logger = logging.getLogger(__name__)


def generate_standardized_id(df):
    """Create unique polyp identifiers for merged datasets.

    The function constructs sequential IDs per original dataset and returns a
    list of new identifiers in ``<dataset>-<number>`` format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``ORIGINAL DATASET`` and ``POLYP ID`` columns.

    Returns
    -------
    list[str]
        Newly generated polyp identifiers corresponding to each row of ``df``.
    """
    # Mapping dictionary for [original_dataset][polyp_id] to numeric identifier
    id_mapping = {}
    new_ids = []

    for idx, row in df.iterrows():
        dataset = row['ORIGINAL DATASET']
        old_polyp_id = row['POLYP ID']

        if dataset not in id_mapping:
            id_mapping[dataset] = {}

        if old_polyp_id not in id_mapping[dataset]:
            id_mapping[dataset][old_polyp_id] = len(id_mapping[dataset]) + 1

        # New polyp_id with dataset prefix and sequential number
        new_polyp_id = f"{dataset}-{id_mapping[dataset][old_polyp_id]:04d}"
        new_ids.append(new_polyp_id)

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
        spreadsheet_path = os.path.join(base_path, dataset, 'annotations.xlsx')
        try:
            df = pd.read_excel(spreadsheet_path)
            df['ORIGINAL DATASET'] = dataset

            # Randomly select 'num_elements' rows if specified and dataset is larger
            if num_elements is not None and len(df) > num_elements:
                df = df.sample(n=num_elements, random_state=random_state)

            dataframes.append(df)
        except FileNotFoundError:
            logger.warning("Spreadsheet not found: %s", spreadsheet_path)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def save_combined_dataframe(combined_df, base_path, filename='annotations.xlsx'):
    output_path = os.path.join(base_path, filename)
    combined_df.to_excel(output_path, index=False)
    logger.info("The combined DataFrame has been saved to %s", output_path)


def validate_combined_dataframe(df: pd.DataFrame, image_dir: str) -> None:
    """Validate merged dataframe integrity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    image_dir : str
        Directory containing the merged image files.

    Raises
    ------
    ValueError
        If required columns are missing or contain null values, or if ``POLYP ID``
        values are not unique.
    FileNotFoundError
        If any ``FRAME`` file referenced in the DataFrame does not exist within
        ``image_dir``.
    """

    required_columns = ["FRAME", "POLYP ID", "BOUNDING BOX"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Null values found in required columns")

    if not df["POLYP ID"].is_unique:
        raise ValueError("Duplicate POLYP ID values detected")

    for frame in df["FRAME"]:
        frame_path = os.path.join(image_dir, frame)
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Referenced image not found: {frame_path}")


def copy_and_rename_images(df, base_path, dest_dir):
    """Copy images from source datasets and assign new unique names.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``ORIGINAL DATASET`` and ``FRAME`` columns.
    base_path : str
        Root directory containing the dataset folders.
    dest_dir : str
        Destination directory where renamed images will be stored.
    """
    os.makedirs(dest_dir, exist_ok=True)  # Ensure the destination directory exists

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        # Construct the path to the source image
        source_image_path = os.path.join(base_path, row['ORIGINAL DATASET'], 'img', row['FRAME'])

        if os.path.exists(source_image_path):
            # Copy the file to the destination directory with its original filename
            new_filename = uuid.uuid4().hex + '.png'
            dest_file_path = os.path.join(dest_dir, new_filename)
            shutil.copy(source_image_path, dest_file_path)
            # Update the DataFrame in place with the new filename
            df.at[index, 'FRAME'] = new_filename
        else:
            logger.warning("File not found: %s", source_image_path)


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
            new_bbox = adjust_bounding_box(row['BOUNDING BOX'], orig_width, orig_height, target_width, target_height,
                                           result_width, result_height)
            df.at[index, 'BOUNDING BOX'] = new_bbox
        else:
            logger.warning("File not found: %s", img_path)


def process_negative_images(source_dir, target_width, target_height):
    """Standardize images in ``source_dir`` without bounding boxes.

    Parameters
    ----------
    source_dir : str
        Directory containing images to be resized.
    target_width : int
        Desired output width.
    target_height : int
        Desired output height.
    """
    if os.path.exists(source_dir):
        files = os.listdir(source_dir)
        for file in tqdm(files, desc="Processing negatives", total=len(files)):
            source_file_path = os.path.join(source_dir, file)
            if os.path.isfile(source_file_path):
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


def merge_datasets(base_path, datasets: list[str], num_elements=None, plot_images=False):
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
    """
    # process_negative_images(base_path, 588, 478)

    combined_df = combine_datasets(base_path, datasets, num_elements=num_elements)
    destination_folder = 'img' if num_elements is None else 'img_' + str(num_elements)
    output_filename = 'metadata.xlsx' if num_elements is None else 'output_' + str(num_elements) + '.xlsx'

    # Keep only entries with a single bounding box (no blank spaces in 'BOUNDING BOX')
    combined_df = combined_df[combined_df['BOUNDING BOX'].str.count(' ') == 0]

    logger.info("Combined dataframe length: %d", len(combined_df))

    # Copy and rename images

    logger.info("COPYING IMAGES")
    copy_and_rename_images(combined_df, base_path, os.path.join(base_path, destination_folder))

    # Calculate target size
    logger.info("CALCULATING TARGET SIZE")
    target_width, target_height = calculate_target_size(combined_df)
    logger.info("Target size: %s x %s", target_width, target_height)

    # Standardize and adjust images and bounding boxes
    logger.info("STANDARDIZING IMAGES")
    process_images_and_bboxes(combined_df, os.path.join(base_path, destination_folder), target_width, target_height)

    if num_elements is None:
        negative_destination_folder = 'negative_img'
        process_negative_images(os.path.join(base_path, negative_destination_folder), target_width, target_height)

    combined_df.drop(columns=['WIDTH', 'HEIGHT'], inplace=True)

    # Standardize polyp ids
    combined_df['POLYP ID'] = generate_standardized_id(combined_df)

    validate_combined_dataframe(combined_df, os.path.join(base_path, destination_folder))

    # Save the combined DataFrame to a spreadsheet
    save_combined_dataframe(combined_df, base_path, output_filename)

    # Loop through the DataFrame and draw bounding boxes on each image
    if plot_images:
        # Create the destination directory if it doesn't exist
        bbox_dest_dir = os.path.join(base_path, 'img_bbox')
        os.makedirs(bbox_dest_dir, exist_ok=True)
        for index, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Drawing bounding boxes"):
            image_filename = row['FRAME']
            input_frame_path = os.path.join(os.path.join(base_path, destination_folder), image_filename)
            output_frame_path = os.path.join(bbox_dest_dir, image_filename)
            draw_bounding_boxes(row['BOUNDING BOX'], input_frame_path, output_frame_path)

    cleanup_temp_artifacts(base_path, destination_folder)
