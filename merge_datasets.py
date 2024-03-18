import argparse
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from common import file_hash


def merge_datasets(base_path, datasets):
    dataframes = []

    for dataset in datasets:
        spreadsheet_path = os.path.join(base_path, dataset, 'annotations.xlsx')
        try:
            df = pd.read_excel(spreadsheet_path)
            df['ORIGINAL DATASET'] = dataset
            dataframes.append(df)
        except FileNotFoundError:
            print(f"Spreadsheet not found: {spreadsheet_path}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def save_combined_dataframe(combined_df, base_path, filename='annotations.xlsx'):
    output_path = os.path.join(base_path, filename)
    combined_df.to_excel(output_path, index=False)
    print(f"The combined DataFrame has been saved to {output_path}")


def copy_and_rename_images(df, base_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)  # Ensure the destination directory exists

    for index, row in df.iterrows():
        # Construct the path to the source image
        source_image_path = os.path.join(base_path, row['ORIGINAL DATASET'], 'img', row['FRAME'])

        if os.path.exists(source_image_path):
            # Copy the file to the destination directory with its original filename
            shutil.copy(source_image_path, dest_dir)
            copied_file_path = os.path.join(dest_dir, row['FRAME'])

            # Compute the file's hash from the destination directory
            new_filename = file_hash(copied_file_path) + '.png'
            new_file_path = os.path.join(dest_dir, new_filename)

            # Rename the file in the destination directory
            os.rename(copied_file_path, new_file_path)

            # Update the DataFrame in place with the new filename
            df.at[index, 'FRAME'] = new_filename

        else:
            print(f"File not found: {source_image_path}")


def calculate_target_size(df):
    """
    Calculate target width and height for image standardization.
    The target width and height are calculated based on the median
    aspect ratio and median dimensions of the dataset.
    :param df: DataFrame containing image metadata
    :return: target width and height as integers
    :rtype: tuple of integers (target_width, target_height)
    """
    # Calculate aspect ratios
    aspect_ratios = df['WIDTH'] / df['HEIGHT']

    # Calculate median aspect ratio
    median_aspect_ratio = np.median(aspect_ratios)

    # Calculate median width and height
    median_width = np.median(df['WIDTH'])
    median_height = np.median(df['HEIGHT'])

    # Determine target dimensions based on the median aspect ratio
    # and median dimensions, rounding to nearest integer
    if median_aspect_ratio > 1:
        # Landscape orientation, fix height and adjust width
        target_height = int(median_height)
        target_width = int(target_height * median_aspect_ratio)
    else:
        # Portrait orientation, fix width and adjust height
        target_width = int(median_width)
        target_height = int(target_width / median_aspect_ratio)

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
    for index, row in df.iterrows():
        img_path = os.path.join(source_dir, row['FRAME'])
        if os.path.exists(img_path):
            orig_width, orig_height, result_width, result_height = standardize_and_override_image(img_path,
                                                                                                  target_width,
                                                                                                  target_height)
            new_bbox = adjust_bounding_box(row['BOUNDING BOX'], orig_width, orig_height, target_width, target_height,
                                           result_width, result_height)
            df.at[index, 'BOUNDING BOX'] = new_bbox
        else:
            print(f"File not found: {img_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Merge datasets into a single DataFrame.')
    parser.add_argument('base_path', type=str, help='The base path where the dataset folders are located.')

    # Parse arguments
    args = parser.parse_args()
    base_path = args.base_path

    datasets = ['piba', 'piccolo', 'sun', 'polar']

    combined_df = merge_datasets(args.base_path, datasets)

    # Keep only entries with a single bounding box (no blank spaces in 'BOUNDING BOX')
    combined_df = combined_df[combined_df['BOUNDING BOX'].str.count(' ') == 0]

    # Copy and rename images
    copy_and_rename_images(combined_df, base_path, os.path.join(base_path, 'img'))

    # Calculate target size
    target_width, target_height = calculate_target_size(combined_df)

    process_images_and_bboxes(combined_df, os.path.join(base_path, 'img'), target_width, target_height)

    # Save the combined DataFrame to a spreadsheet
    save_combined_dataframe(combined_df, base_path, 'output.xlsx')


if __name__ == "__main__":
    main()
