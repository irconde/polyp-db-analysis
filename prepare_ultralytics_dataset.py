import pandas as pd
import os
from shutil import copy2
from PIL import Image
import numpy as np
import argparse
import yaml

# Function to convert bounding box from absolute to YOLO format
def convert_bbox_to_yolo_format(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Normalize
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height


def main(base_path):
    # Load the Excel file
    excel_path = os.path.join(base_path, 'annotations.xlsx')
    df = pd.read_excel(excel_path)

    # Shuffle the DataFrame rows
    df = df.sample(frac=1).reset_index(drop=True)

    # Define your class labels and corresponding IDs
    class_labels_to_id = {'IIa': 0, 'IIb': 1, 'IIc': 2, 'III': 3,
                          'Ip': 4, 'Is': 5, 'Isp': 6}

    source_dir = os.path.join(base_path, 'img/')
    base_dir = os.path.join(base_path, 'dataset/')
    train_dir = os.path.join(base_dir, 'images/train/')
    val_dir = os.path.join(base_dir, 'images/val/')
    test_dir = os.path.join(base_dir, 'images/test/')

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir.replace('images', 'labels'), exist_ok=True)
    os.makedirs(val_dir.replace('images', 'labels'), exist_ok=True)
    os.makedirs(test_dir.replace('images', 'labels'), exist_ok=True)

    # Dataset split proportions
    train_split = 0.7
    val_split = 0.2
    # test_split = 0.1  # Remainder for test set

    # Calculate split indices
    num_images = len(df)
    num_train = int(num_images * train_split)
    num_val = int(num_images * val_split)
    num_test = num_images - num_train - num_val

    # data for yaml dataset configuration file
    dataset_config = {
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,  # If you have a test set
        'nc': len(class_labels_to_id),
        'names': list(class_labels_to_id.keys())
    }

    # Process and split dataset
    for index, row in df.iterrows():
        # Get image dimensions
        image_path = os.path.join(source_dir, row['FRAME'])
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Convert bounding box to YOLO format
        bbox = [float(x) for x in
                row['BOUNDING BOX'].split(',')]  # Assuming bbox is stored as "x_min,y_min,x_max,y_max"
        x_center_norm, y_center_norm, width_norm, height_norm = convert_bbox_to_yolo_format(bbox, image_width,
                                                                                            image_height)

        # Determine the class ID
        class_id = class_labels_to_id[row['SHAPE']]

        # Prepare the annotation line
        annotation_line = f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"

        # Assign to train, val, or test based on the split
        if index < num_train:
            dest_dir = train_dir
        elif index < num_train + num_val:
            dest_dir = val_dir
        else:
            dest_dir = test_dir

        dest_image_path = os.path.join(dest_dir, row['FRAME'])
        dest_label_path = dest_image_path.replace('images', 'labels').replace('.png', '.txt')

        # Copy image to the corresponding directory
        copy2(image_path, dest_image_path)
    
        # Write annotation to the corresponding file
        with open(dest_label_path, 'w') as f:
            f.write(annotation_line)

    # Path for the YAML file
    yaml_path = os.path.join(base_dir, 'data.yaml')

    # Writing the dataset configuration to a YAML file
    with open(yaml_path, 'w') as yamlfile:
        documents = yaml.dump(dataset_config, yamlfile)



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Prepare dataset for Ultralytics YOLO training.')
    parser.add_argument('base_path', type=str, help='Base path where the Excel and img folder are located')
    args = parser.parse_args()
    main(args.base_path)