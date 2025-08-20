"""Utilities for counting class occurrences in YOLO-style datasets."""

import os
from collections import defaultdict


def count_samples(root_folder):
    """Count samples per class in YOLO label files.

    Parameters
    ----------
    root_folder : str
        Path to the dataset containing ``train``, ``val`` and ``test``
        subdirectories with label files.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapping of split names to class counts.
    """
    # Initialize a dictionary to keep track of the class counts for each split
    class_counts = {
        'test': defaultdict(int),
        'train': defaultdict(int),
        'val': defaultdict(int)
    }

    # Define the subfolders
    subfolders = ['test', 'train', 'val']

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)

        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)

                    with open(file_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            class_id = line.split()[0]
                            class_counts[subfolder][class_id] += 1

    return class_counts
