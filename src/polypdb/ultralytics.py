import pandas as pd
import os
from shutil import copy2
from PIL import Image
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from .utils import resize_with_padding, normalize_image


# Function to load configuration params from yaml figuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# Function to split data according to a given configuration
def split_data(df, config):
    # Extract splitting options for easier access
    splitting_options = config['splitting_options']
    train_ratio = splitting_options['train_ratio']
    val_ratio = splitting_options['validation_ratio']
    test_ratio = splitting_options['test_ratio']
    split_type = splitting_options['split_type']
    split_level = splitting_options['split_level']

    # Determine if stratification is requested
    stratify_by_shape = (split_type == 'STRATIFIED')

    # Call the appropriate function based on the split level
    if split_level == 'POLYP ID':
        return polyp_level_split(df, (train_ratio, val_ratio, test_ratio), stratify_by_shape)
    elif split_level == 'FRAME':
        return image_level_split(df, (train_ratio, val_ratio, test_ratio), stratify_by_shape)
    else:
        raise ValueError("Unsupported split level provided in configuration.")


# id_col='POLYP ID', stratify_col='SHAPE', ratios=(0.7, 0.2, 0.1)
def stratified_split_if_possible(df, id_col, stratify_col, ratios, random_state=42):
    # Compute the minimum number of samples needed to split according to the smallest ratio
    min_samples = int(1 / min(ratios))
    # Determine the number of splits based on non-zero ratios
    num_splits = sum(ratio > 0 for ratio in ratios)

    # Initialize dictionaries to store the split dataframes
    train_dfs = []
    val_dfs = []
    test_dfs = []

    # Randomize the order of shapes to avoid any inherent ordering bias
    shapes = df[stratify_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(shapes)

    # Check for each shape whether stratified splitting is possible
    for shape in df[stratify_col].unique():
        shape_data = df[df[stratify_col] == shape]
        polyps = shape_data[id_col].unique()
        polyp_count = len(polyps)

        if polyp_count < num_splits:
            raise ValueError(f"Not enough polyps for shape '{shape}'. At least 3 polyps are required.")

        if polyp_count >= min_samples:
            # Perform stratified split
            if ratios[2] > 0:
                train_val_polyps, test_polyps = train_test_split(polyps, test_size=ratios[2], random_state=random_state)
            else:
                train_val_polyps, test_polyps = polyps, []

            train_polyps, val_polyps = train_test_split(train_val_polyps, test_size=ratios[1] / (ratios[0] + ratios[1]),
                                                        random_state=random_state)

            # Append DataFrames by selecting data for each polyp ID
            train_dfs.append(df[df[id_col].isin(train_polyps)])
            val_dfs.append(df[df[id_col].isin(val_polyps)])
            if len(test_polyps) > 0:
                test_dfs.append(df[df[id_col].isin(test_polyps)])

        else:
            # Evenly distribute polyps across splits if stratified split is not possible
            train_set, val_set, test_set = _evenly_distribute_polyps(df, polyps, id_col, ratios, num_splits, random_state)
            train_dfs.append(train_set)
            val_dfs.append(val_set)
            if not test_set.empty:
                test_dfs.append(test_set)

    # Concatenate all dataframes for each split
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    return train_df, val_df, test_df


def _evenly_distribute_polyps(df, polyps, id_col, ratios, num_splits, random_state):
    np.random.seed(random_state)
    np.random.shuffle(polyps)

    total_polyps = len(polyps)

    # Compute the initial number of polyps for each split based on the provided ratios
    n_train = max(1, round(total_polyps * ratios[0]))
    n_val = max(1, round(total_polyps * ratios[1]))
    n_test = max(1, total_polyps - n_train - n_val) if ratios[2] > 0 else 0

    min_test = 0 if  ratios[2] == 0 else 1

    # Adjust if the sum exceeds the available polyps
    while n_train + n_val + n_test > total_polyps:
        if n_test > min_test:
            n_test = max(min_test, n_test - 1)
        elif n_val > 1:
            n_val = max(1, n_val - 1)
        else:
            n_train = max(1, n_train - 1)

    # Ensure each split gets at least one polyp if the total polyps are greater than or equal to the number of splits
    if total_polyps >= num_splits:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_test) if ratios[2] > 0 else 0

    # Create splits by selecting data for each polyp ID
    train_set = df[df[id_col].isin(polyps[:n_train])]
    val_set = df[df[id_col].isin(polyps[n_train:n_train + n_val])]
    test_set = df[df[id_col].isin(polyps[n_train + n_val:])] if n_test > 0 else pd.DataFrame()

    return train_set, val_set, test_set


def split_by_field(df, group_field, ratios, random_state=42):
    # Group data by the specified field
    train_ratio, validation_ratio, test_ratio = ratios
    grouped = df.groupby(group_field)
    group_keys = list(grouped.groups.keys())
    np.random.shuffle(group_keys)  # Shuffle to avoid any inherent ordering bias

    # Split group_keys into train and remaining sets
    train_keys, remaining_keys = train_test_split(group_keys, test_size=1 - train_ratio, stratify=None,
                                                  random_state=random_state)

    if test_ratio == 0:
        val_df = pd.concat([grouped.get_group(key) for key in remaining_keys])
        test_df = pd.DataFrame()
    else:
        # Calculate the relative ratio for remaining splits
        remaining_ratio = validation_ratio / (validation_ratio + test_ratio)
        # Split remaining_keys into validation and test sets
        val_keys, test_keys = train_test_split(remaining_keys, test_size=1 - remaining_ratio, stratify=None,
                                               random_state=random_state)
        val_df = pd.concat([grouped.get_group(key) for key in val_keys])
        test_df = pd.concat([grouped.get_group(key) for key in test_keys])

    # Collect data for each split
    train_df = pd.concat([grouped.get_group(key) for key in train_keys])

    return train_df, val_df, test_df


# Function to carry out data splitting at the polyp level
def polyp_level_split(df, ratios, stratify_by_shape=False):
    random_state = 42
    train_ratio, validation_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0, atol=0.01):
        raise ValueError("The sum of train, validation, and test ratios must be close to 1")

    if stratify_by_shape:
        train_df, val_df, test_df = stratified_split_if_possible(df, 'POLYP ID', 'SHAPE',
                                                                 (train_ratio, validation_ratio, test_ratio),
                                                                 random_state)
    else:
        train_df, val_df, test_df = split_by_field(df, 'POLYP ID', ratios, random_state)

    return train_df, val_df, test_df


# Function to carry out data splitting at the frame level
def image_level_split(df, ratios, stratify_by_shape=False):
    train_ratio, validation_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0, atol=0.01):
        raise ValueError("The sum of train, validation, and test ratios must be close to 1")

        # Check if stratification is requested and feasible
    if stratify_by_shape:
        if df['SHAPE'].nunique() < 2 or any(df['SHAPE'].value_counts() < 2):
            raise ValueError("Stratification by shape not possible due to insufficient data per class.")

        # Stratification column
    stratify_col = df['SHAPE'] if stratify_by_shape else None

    # Splitting dataset into train and remaining sets
    train_df, remaining_df = train_test_split(
        df, test_size=1 - train_ratio,
        stratify=stratify_col if stratify_col is not None else None
    )

    # Handling zero test ratio
    if test_ratio == 0:
        val_df = remaining_df
        test_df = pd.DataFrame()  # Return an empty DataFrame if no test data is needed
    else:
        # Calculate the relative ratio for remaining splits
        remaining_ratio = validation_ratio / (validation_ratio + test_ratio)

        # Split remaining data into validation and test sets
        val_df, test_df = train_test_split(
            remaining_df, test_size=1 - remaining_ratio,
            stratify=stratify_col.loc[remaining_df.index] if stratify_col is not None else None
        )

    return train_df, val_df, test_df


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

    # Clamp the values to [0, 1] to avoid out-of-bounds errors
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def create_folder_hierarchy(save_path, create_test_dir=False):
    train_dir = os.path.join(save_path, 'images/train/')
    val_dir = os.path.join(save_path, 'images/val/')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_dir.replace('images', 'labels'), exist_ok=True)
    os.makedirs(val_dir.replace('images', 'labels'), exist_ok=True)
    test_dir = None
    if create_test_dir:
        test_dir = os.path.join(save_path, 'images/test/')
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(test_dir.replace('images', 'labels'), exist_ok=True)
    return train_dir, val_dir, test_dir


def process_img(train_df, val_df, test_df, classes, source_path, save_path):
    # Process and split dataset into directories
    for subset, data in zip(('train', 'val', 'test'), (train_df, val_df, test_df)):
        if data is not None and not data.empty:
            dest_dir = os.path.join(save_path, f'images/{subset}/')
            for index, row in data.iterrows():
                image_path = os.path.join(source_path, row['FRAME'])
                dest_image_path = os.path.join(dest_dir, row['FRAME'])
                dest_label_path = dest_image_path.replace('images', 'labels').replace('.png', '.txt')

                # Copy image to the corresponding directory
                copy2(image_path, dest_image_path)
                # Get image dimensions
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
                # Convert bounding box to YOLO format
                bbox = [float(x) for x in
                        row['BOUNDING BOX'].split(',')]  # Assuming bbox is stored as "x_min,y_min,x_max,y_max"
                x_center_norm, y_center_norm, width_norm, height_norm = convert_bbox_to_yolo_format(bbox,
                                                                                                    image_width,
                                                                                                    image_height)
                # Determine the class ID
                class_id = 0 if len(classes) == 1 else classes[row['SHAPE']]

                # Prepare the annotation line
                annotation_line = f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
                # Write annotation to the corresponding file
                with open(dest_label_path, 'w') as f:
                    f.write(annotation_line)


def save_dataset_config_file(save_path, train_path, val_path, test_path, classes):
    # Path for the YAML configuration file to be used in training
    yaml_path = os.path.join(save_path, 'data.yaml')
    # data for yaml dataset configuration file
    dataset_config = {
        'train': train_path,
        'val': val_path,
        'nc': len(classes),
        'names': list(classes.keys())
    }
    if test_path is not None:
        dataset_config['test'] = test_path

    # Writing the dataset configuration to a YAML file
    with open(yaml_path, 'w') as yamlfile:
        yaml.dump(dataset_config, yamlfile)


def calculate_negative_split_sizes(sizes, config):
    negative_ratio = config['ratio']
    distribution = config['distribution']
    total_positives = sum(sizes)
    if total_positives == 0:
        raise ValueError("Total size of splits must be greater than zero to allocate negative samples.")

    if distribution == "SPLIT-RATIO":
        proportions = [size / total_positives for size in sizes]
    elif distribution == "EVEN":
        non_zero_count = sum(1 for size in sizes if size > 0)
        proportions = [1 / non_zero_count if size > 0 else 0 for size in sizes]
    else:
        raise ValueError("Unsupported distribution type specified.")

    total_negatives = int(total_positives * negative_ratio)
    neg_sizes = [int(total_positives * prop * negative_ratio) for prop in proportions]
    rounding_correction = total_negatives - sum(neg_sizes)
    # Apply rounding correction to the last non-zero split
    if rounding_correction != 0:
        for i in reversed(range(len(neg_sizes))):
            if sizes[i] > 0:  # Ensure the split is non-zero
                neg_sizes[i] += rounding_correction
                break

    return neg_sizes


def split_negative_data(negative_sample_path, split_sizes):
    negative_img_paths = [os.path.join(negative_sample_path, f) for f in os.listdir(negative_sample_path) if
                          f.endswith('.jpg')]
    total_samples = sum(split_sizes)
    total_img_count = len(negative_img_paths)
    if total_samples > total_img_count:
        raise ValueError("Not enough negative samples to fulfill the requested split sizes.")

    np.random.shuffle(negative_img_paths)  # Shuffle to ensure randomness

    train_end = split_sizes[0]
    val_end = train_end + split_sizes[1]

    neg_train_split = negative_img_paths[:train_end]
    neg_val_split = negative_img_paths[train_end:val_end]
    neg_test_split = negative_img_paths[val_end:val_end + split_sizes[2]]

    return [neg_train_split, neg_val_split, neg_test_split]


def process_negative_samples(negative_splits, save_path):
    split_types = ['train', 'val', 'test']
    for index, split in enumerate(negative_splits):
        # Corresponding folder for the split
        split_type = split_types[index]
        img_dest_dir = os.path.join(save_path, f'images/{split_type}/')
        label_dest_dir = os.path.join(save_path, f'labels/{split_type}/')
        for img_path in split:
            # Copy image to the corresponding directory
            filename = os.path.basename(img_path)
            dest_path = os.path.join(img_dest_dir, filename)
            copy2(img_path, dest_path)
            # Create an empty annotation file
            label_path = os.path.join(label_dest_dir, filename.replace('.jpg', '.txt'))
            open(label_path, 'w').close()  # Create an empty file for YOLO


def _save_cropped_image(image_path, bbox, dest_path, output_size=(224, 224)):
    """Crop ``image_path`` using ``bbox``, resize and normalize, then save to ``dest_path``.

    The crop is resized to ``output_size`` while preserving aspect ratio via padding.
    ImageNet statistics are used for normalization.
    """
    with Image.open(image_path) as img:
        x_min, y_min, x_max, y_max = [int(float(v)) for v in bbox.split(',')]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.width, x_max)
        y_max = min(img.height, y_max)
        cropped = img.crop((x_min, y_min, x_max, y_max)).convert('RGB')

        # Resize with padding to maintain aspect ratio
        resized = resize_with_padding(cropped, target_size=output_size)

        # Apply per-channel normalization
        normalized = normalize_image(resized)

        # Convert back to uint8 for saving
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        unnormalized = np.clip((normalized * std + mean) * 255, 0, 255).astype(np.uint8)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        Image.fromarray(unnormalized).save(dest_path, format='JPEG')


def process_classification_images(train_df, val_df, test_df, source_path, save_path,
                                  classification_type, ng_img_src_path=None, negative_cfg=None):
    """Create a HyperKvasir-style dataset for classification."""
    root = os.path.join(save_path, 'labeled-images')
    for subset, data in zip(('train', 'val', 'test'), (train_df, val_df, test_df)):
        if data is not None and not data.empty:
            for _, row in data.iterrows():
                img_path = os.path.join(source_path, row['FRAME'])
                cls = 'polyp' if classification_type == 'BINARY' else row['SHAPE']
                dest_dir = os.path.join(root, subset, 'pathological-findings', cls)
                dest_name = os.path.splitext(row['FRAME'])[0] + '.jpg'
                dest_path = os.path.join(dest_dir, dest_name)
                _save_cropped_image(img_path, row['BOUNDING BOX'], dest_path)

    if classification_type == 'BINARY' and ng_img_src_path is not None and negative_cfg is not None:
        neg_sizes = calculate_negative_split_sizes(
            [len(train_df), len(val_df), len(test_df)], negative_cfg
        )
        neg_splits = split_negative_data(ng_img_src_path, neg_sizes)
        for subset, split in zip(('train', 'val', 'test'), neg_splits):
            dest_dir = os.path.join(root, subset, 'pathological-findings', 'negative')
            os.makedirs(dest_dir, exist_ok=True)
            for img_path in split:
                copy2(img_path, os.path.join(dest_dir, os.path.basename(img_path)))


def prepare_dataset(config_file):
    """Prepare dataset for Ultralytics YOLO training based on a YAML configuration."""
    # Load the configuration options
    config = load_config(config_file)
    src_path = config['src_path']

    # Define class labels and corresponding IDs
    if config['classification_type'] == 'BINARY':
        classes = {'polyp': 0}
    else:
        classes = {'IIa': 0, 'IIb': 1, 'IIc': 2, 'III': 3,
                   'Ip': 4, 'Is': 5, 'Isp': 6}

    img_src_path = os.path.join(src_path, 'img/')
    ng_img_src_path = os.path.join(src_path, 'negative_img/')
    save_path = os.path.dirname(os.path.abspath(config_file))

    # Load the Excel file
    excel_path = os.path.join(src_path, 'metadata.xlsx')
    df = pd.read_excel(excel_path)

    # Split the data according to the configuration
    train_df, val_df, test_df = split_data(df, config)

    print(train_df.groupby('SHAPE')['POLYP ID'].count())
    print(val_df.groupby('SHAPE')['POLYP ID'].count())
    print(test_df.groupby('SHAPE')['POLYP ID'].count())

    # Create folder hierarchy
    train_path, val_path, test_path = create_folder_hierarchy(save_path, len(test_df.index) > 0)

    # Split images accordingly
    process_img(train_df, val_df, test_df, classes, img_src_path, save_path)

    # Process negative samples
    neg_split_sizes = calculate_negative_split_sizes([len(train_df), len(val_df), len(test_df)],
                                                     config['negative_sampling'])
    neg_splits = split_negative_data(ng_img_src_path, neg_split_sizes)
    process_negative_samples(neg_splits, save_path)

    # Save dataset summary in configuration file
    save_dataset_config_file(save_path, train_path, val_path, test_path, classes)


def prepare_classification_dataset(config_file):
    """Prepare a HyperKvasir-style dataset for classification."""
    config = load_config(config_file)
    src_path = config['src_path']

    if config['classification_type'] == 'BINARY':
        classes = ['polyp', 'negative']
    else:
        classes = ['IIa', 'IIb', 'IIc', 'III', 'Ip', 'Is', 'Isp']

    img_src_path = os.path.join(src_path, 'img/')
    ng_img_src_path = os.path.join(src_path, 'negative_img/')
    save_path = os.path.dirname(os.path.abspath(config_file))

    df = pd.read_excel(os.path.join(src_path, 'metadata.xlsx'))

    train_df, val_df, test_df = split_data(df, config)

    root = os.path.join(save_path, 'labeled-images')
    for subset in ('train', 'val', 'test'):
        for cls in classes:
            os.makedirs(os.path.join(root, subset, 'pathological-findings', cls), exist_ok=True)

    process_classification_images(
        train_df,
        val_df,
        test_df,
        img_src_path,
        save_path,
        config['classification_type'],
        ng_img_src_path,
        config.get('negative_sampling'),
    )

