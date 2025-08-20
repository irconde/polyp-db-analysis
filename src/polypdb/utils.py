import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def get_image_size(image_path):
    """Return the width and height of an image.

    The image is opened using :mod:`PIL.Image` instead of OpenCV.

    Args:
        image_path: Path to the image.

    Returns:
        Tuple ``(width, height)`` representing the size of the image.
    """
    with Image.open(image_path) as img:
        return img.width, img.height


def crop_image_padding(
    img_path,
    cropped_img_path,
    mask_path=None,
    cropped_mask_path=None,
    padding_color=(0, 0, 0),
    threshold=10,
):
    """Crop an image and optional mask, stripping away padding pixels.

    ``img_path`` is validated before attempting to open it.  Passing a non
    existing path (or an invalid file descriptor) previously resulted in a
    rather opaque ``OSError: [Errno 9] Bad file descriptor`` bubbling up from
    :func:`PIL.Image.open`.  By explicitly checking for path existence we can
    raise a clearer ``FileNotFoundError`` and avoid confusing descriptor
    errors.

    Parameters
    ----------
    img_path:
        Location of the input image to be cropped.
    cropped_img_path:
        Output location where the cropped image will be written.
    mask_path:
        Optional path to a segmentation mask corresponding to ``img_path``.
    cropped_mask_path:
        Destination for the cropped mask when ``mask_path`` is provided.
    padding_color:
        RGB colour considered as padding and removed from the image.
    threshold:
        Permissible channel deviation from ``padding_color`` to still be
        regarded as padding.
    """

    img_path = os.fspath(img_path)
    cropped_img_path = os.fspath(cropped_img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    with Image.open(img_path) as img:
        arr = np.array(img)

        # Optimization: use NumPy to locate the bounding box of non-padding pixels
        padding = np.array(padding_color, dtype=arr.dtype)
        diff = np.abs(arr - padding)
        mask = np.all(diff <= threshold, axis=-1)
        non_empty_rows = np.where(~mask.all(axis=1))[0]
        non_empty_cols = np.where(~mask.all(axis=0))[0]

        if len(non_empty_rows) == 0 or len(non_empty_cols) == 0:
            left, upper, right, lower = 0, 0, img.width, img.height
        else:
            upper, lower = non_empty_rows[0], non_empty_rows[-1] + 1
            left, right = non_empty_cols[0], non_empty_cols[-1] + 1

        # Crop the image to the bounding box to remove padding
        cropped_img = img.crop((left, upper, right, lower))
        os.makedirs(os.path.dirname(cropped_img_path) or ".", exist_ok=True)
        with open(cropped_img_path, "wb") as f:
            cropped_img.save(f)

    if mask_path is not None:
        mask_path = os.fspath(mask_path)
        cropped_mask_path = os.fspath(cropped_mask_path)
        if os.path.exists(mask_path):
            with Image.open(mask_path) as mask_img:
                # Crop the mask to the same bounding box
                cropped_mask_img = mask_img.crop((left, upper, right, lower))
                os.makedirs(os.path.dirname(cropped_mask_path) or ".", exist_ok=True)
                with open(cropped_mask_path, "wb") as f:
                    cropped_mask_img.save(f)


def resize_and_maintain_aspect(img_path, output_path, target_size=(854, 480), fill_color=(0, 0, 0)):
    """
    Resize an image to a target size maintaining the aspect ratio.
    If the image is smaller than the target, pad it with a specified color.
    :param img_path: Path to the input image.
    :param output_path: Path to the output image.
    :param target_size: Tuple (width, height) of the target size.
    :param fill_color: Tuple (R, G, B) of the color to fill the padded area.
    :return: None. The image is saved to the output path.
    """
    with Image.open(img_path) as img:
        # Maintain aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # If the image is smaller than the target, pad it
        if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
            new_img = Image.new("RGB", target_size, fill_color)
            new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                                (target_size[1] - img.size[1]) // 2))
        else:
            new_img = img

        new_img.save(output_path)


def resize_with_padding(img, target_size=(224, 224), fill_color=(0, 0, 0)):
    """Resize ``img`` to ``target_size`` maintaining aspect ratio by padding.

    :param img: ``PIL.Image`` to be resized.
    :param target_size: Desired output size ``(width, height)``.
    :param fill_color: RGB color tuple used for padding.
    :return: Resized and padded ``PIL.Image``.
    """
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, fill_color)
    new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                         (target_size[1] - img.size[1]) // 2))
    return new_img


def normalize_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize ``img`` using per-channel ``mean`` and ``std``.

    The image is expected to be in RGB format.

    :param img: ``PIL.Image`` to normalize.
    :param mean: Tuple of channel means.
    :param std: Tuple of channel standard deviations.
    :return: ``numpy.ndarray`` containing the normalized image data.
    """
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)
    return (arr - mean_arr) / std_arr


def adjust_bounding_box(original_box, original_size, resized_size, padding):
    """
    Adjusts bounding box coordinates based on the resized image dimensions and padding.
    :param original_box: Tuple (x1, y1, x2, y2) of the original bounding box coordinates.
    :param original_size: Tuple (width, height) of the original image size.
    :param resized_size: Tuple (width, height) of the resized image dimensions.
    :param padding: Tuple (left_padding, top_padding) indicating the amount of padding added.
    :return: Tuple (x1, y1, x2, y2) of the adjusted bounding box coordinates.
    """

    scale_x = resized_size[0] / original_size[0]
    scale_y = resized_size[1] / original_size[1]

    # Scale the original bounding box coordinates
    x1, y1, x2, y2 = original_box
    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    # Adjust for padding
    x1_adjusted = x1_scaled + padding[0]
    y1_adjusted = y1_scaled + padding[1]
    x2_adjusted = x2_scaled + padding[0]
    y2_adjusted = y2_scaled + padding[1]

    return x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted


def draw_bounding_boxes(bounding_boxes, image_path, save_path):
    """Overlay ``bounding_boxes`` on ``image_path`` and save to ``save_path``.

    Parameters
    ----------
    bounding_boxes : str
        Comma-separated bounding box coordinates in ``x1,y1,x2,y2`` format.
    image_path : str
        Path to the input image.
    save_path : str
        Location where the annotated image will be written.
    """
    # Open the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        coordinates = [int(coord.strip()) for coord in bounding_boxes.split(',') if coord.strip()]
        draw.rectangle(coordinates, outline="red", width=2)

        # Save the image with bounding boxes
        img.save(save_path)


def move_images(source_dir, dest_dir, df, filename_col='FRAME'):
    """
    Move images from a source directory to a destination directory based on filenames in a DataFrame.

    Parameters:
    - source_dir: The source directory where images are currently stored.
    - dest_dir: The destination directory where images should be moved.
    - df: The DataFrame containing the filenames of the images to be moved.
    - filename_col: The name of the DataFrame column containing the image filenames. Defaults to 'FRAME'.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        frame_filename = row[filename_col]
        source_path = os.path.join(source_dir, frame_filename)
        destination_path = os.path.join(dest_dir, frame_filename)

        # Attempt to move the file
        try:
            shutil.move(source_path, destination_path)
        except IOError as e:
            logger.error("Could not move %s. Error: %s", frame_filename, e)


def move_images_parallel(
    source_dir,
    dest_dir,
    df,
    filename_col="FRAME",
    source_path_col=None,
    copy=False,
):
    """Move images concurrently using :class:`ThreadPoolExecutor`.

    Parameters
    ----------
    source_dir:
        Base directory where images are stored when ``source_path_col`` is
        ``None``.
    dest_dir:
        Directory where images will be placed.
    df:
        DataFrame containing image file information.
    filename_col:
        Column in ``df`` with the destination image filename.
    source_path_col:
        Optional column in ``df`` with full source paths. When provided,
        ``source_dir`` is ignored for that row.
    copy:
        If ``True``, copy images instead of moving them.
    """

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
        except IOError as e:
            logger.error("Could not %s %s. Error: %s", action, frame_filename, e)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(_transfer, rows), total=total, desc=f"{action.title()}ing images"))
    logger.info("Finished %s images", action)


def validate_conversion(dest_dir, df, filename_col='FRAME'):
    """Validate that moved images match entries in ``df``.

    The function counts the rows in ``df`` and compares them against the
    number of image files located in ``dest_dir/img``. If the counts do not
    match, detailed diagnostics about missing, unexpected or duplicated file
    names are logged and a :class:`ValueError` is raised.

    Parameters
    ----------
    dest_dir:
        Root directory of the converted dataset. An ``img`` sub-directory is
        expected to contain the image files.
    df:
        DataFrame describing the dataset.
    filename_col:
        Name of the column in ``df`` containing the image filenames.
    """

    img_dir = os.path.join(dest_dir, 'img')
    try:
        files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    except FileNotFoundError:
        logger.error("Image directory %s does not exist", img_dir)
        raise

    df_count = len(df)
    file_count = len(files)

    duplicates = df[df[filename_col].duplicated()][filename_col].tolist()
    if duplicates:
        logger.error("Duplicated filenames in metadata: %s", ", ".join(sorted(duplicates)))

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
