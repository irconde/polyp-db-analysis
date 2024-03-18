import argparse
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from common import file_hash, draw_bounding_boxes


def merge_datasets(base_path, datasets, num_elements=None):
    """
    Merge datasets into a single DataFrame, optionally including a specified number of elements from each dataset.

    Args:
    - base_path (str): The base path where the dataset folders are located.
    - datasets (list of str): A list of dataset names to merge.
    - num_elements (int, optional): The number of elements to include from each dataset.
                                    If None (default), all rows are included.

    Returns:
    - pd.DataFrame: The combined DataFrame with an 'ORIGINAL DATASET' column indicating the source dataset
                    for each row. If num_elements is specified and is less than the number of available rows
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
                df = df.sample(n=num_elements)

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


def main(base_path, num_elements, plot_images):

    datasets = ['piba', 'piccolo', 'sun', 'polar']

    combined_df = merge_datasets(args.base_path, datasets, num_elements=num_elements)
    destination_folder = 'img' if num_elements is None else 'img_'+str(num_elements)
    output_filename = 'output.xlsx' if num_elements is None else 'output_'+str(num_elements)+'.xlsx'

    # Keep only entries with a single bounding box (no blank spaces in 'BOUNDING BOX')
    combined_df = combined_df[combined_df['BOUNDING BOX'].str.count(' ') == 0]

    # Copy and rename images
    print("COPYING IMAGES")
    copy_and_rename_images(combined_df, base_path, os.path.join(base_path, destination_folder))

    # Calculate target size
    print("CALCULATING TARGET SIZE")
    target_width, target_height = calculate_target_size(combined_df)
    print('Target size: ', target_width, ' x ', target_height)

    # Standardize and adjust images and bounding boxes
    print("STANDARDIZING IMAGES")
    process_images_and_bboxes(combined_df, os.path.join(base_path, destination_folder), target_width, target_height)

    # Save the combined DataFrame to a spreadsheet
    save_combined_dataframe(combined_df, base_path, output_filename)

    # Loop through the DataFrame and draw bounding boxes on each image
    if plot_images:
        # Create the destination directory if it doesn't exist
        bbox_dest_dir = os.path.join(base_path, 'img_bbox')
        os.makedirs(bbox_dest_dir, exist_ok=True)
        for index, row in combined_df.iterrows():
            image_filename = row['FRAME']
            input_frame_path = os.path.join(os.path.join(base_path, destination_folder), image_filename)
            output_frame_path = os.path.join(bbox_dest_dir, image_filename)
            draw_bounding_boxes(row['BOUNDING BOX'], input_frame_path, output_frame_path)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Merge datasets into a single DataFrame.')
    # Add arguments
    parser.add_argument('base_path', type=str, help='The base path where the dataset folders are located.')
    parser.add_argument("--num_elements", type=int, default=None,
                        help="Number of elements to include from each dataset. If not specified, all rows are included.")
    parser.add_argument("--plot", action='store_true', help="Enable image plotting.")

    # Parse arguments
    args = parser.parse_args()

    main(args.base_path, args.num_elements, args.plot)
