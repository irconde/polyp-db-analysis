import os
import shutil

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np


def get_image_size(image_path):
    """
    Returns the width and height of an image.
    Args:
        image_path: Path to the image.
    Returns:
        width: Width of the image.
        height: Height of the image.
    """
    import cv2
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    return width, height


def crop_image_padding(img_path, cropped_img_path, mask_path=None, cropped_mask_path=None, padding_color=(0, 0, 0),
                       threshold=10):
    """
    Crops the image and mask to remove any padding.
    Args:
        img_path: Path to the input image.
        cropped_img_path: Path to save the cropped image.
        mask_path: Path to the input mask.
        cropped_mask_path: Path to save the cropped mask.
        padding_color: Color to check for padding.
        threshold: Threshold for color difference.
    Returns:
        None. Saves the cropped image and mask to the specified paths.
    """
    with Image.open(img_path) as img:
        pixels = img.load()

        def is_close_to_color(pixel, color, threshold):
            return all(abs(p - c) <= threshold for p, c in zip(pixel, color))

        # Check for left padding
        left_padding = 0
        for x in range(img.width):
            if all(is_close_to_color(pixels[x, y], padding_color, threshold) for y in range(img.height)):
                left_padding += 1
            else:
                break

        # Check for right padding
        right_padding = 0
        for x in range(img.width - 1, -1, -1):
            if all(is_close_to_color(pixels[x, y], padding_color, threshold) for y in range(img.height)):
                right_padding += 1
            else:
                break

        # Check for top padding
        top_padding = 0
        for y in range(img.height):
            if all(is_close_to_color(pixels[x, y], padding_color, threshold) for x in range(img.width)):
                top_padding += 1
            else:
                break

        # Check for bottom padding
        bottom_padding = 0
        for y in range(img.height - 1, -1, -1):
            if all(is_close_to_color(pixels[x, y], padding_color, threshold) for x in range(img.width)):
                bottom_padding += 1
            else:
                break

        # Calculate the bounding box of the non-padding area
        left = left_padding
        upper = top_padding
        right = img.width - right_padding
        lower = img.height - bottom_padding

        # Crop the image to the bounding box to remove padding
        cropped_img = img.crop((left, upper, right, lower))
        cropped_img.save(cropped_img_path)

        if mask_path is not None:
            with Image.open(mask_path) as mask_img:
                # Crop the mask to the bounding box to remove padding
                cropped_mask_img = mask_img.crop((left, upper, right, lower))
                cropped_mask_img.save(cropped_mask_path)


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
            print(f"Could not move {frame_filename}. Error: {e}")
