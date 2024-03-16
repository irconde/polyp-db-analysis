import cv2
import numpy as np
from PIL import Image
import hashlib
import os
import shutil


# Function to calculate image properties
def get_image_properties(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Calculating aspect ratio
    aspect_ratio = width / height if height != 0 else 0

    # Calculating average intensity per channel (Red, Green, Blue)
    avg_intensity_per_channel = np.mean(image, axis=(0, 1))
    avg_red_intensity = avg_intensity_per_channel[0]
    avg_green_intensity = avg_intensity_per_channel[1]
    avg_blue_intensity = avg_intensity_per_channel[2]

    # Calculating brightness and contrast
    brightness = np.mean(avg_intensity_per_channel)
    contrast = np.std(image)

    # Blurriness
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    return width, height, aspect_ratio, avg_red_intensity, avg_green_intensity, avg_blue_intensity, brightness, contrast, laplacian_var


def get_image_size(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    return width, height


def crop_image_padding(img_path, cropped_img_path, mask_path=None, cropped_mask_path=None, padding_color=(0, 0, 0),
                       threshold=10):
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


def file_hash(filepath):
    """Generate a hash for a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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

    return (x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted)
