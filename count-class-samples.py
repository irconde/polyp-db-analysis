import os
import sys
from collections import defaultdict

def count_samples(root_folder):
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <root_folder>")
        sys.exit(1)

    root_folder = sys.argv[1]
    class_counts = count_samples(root_folder)

    for split in ['test', 'train', 'val']:
        print(f"\n{split.capitalize()} split:")
        for class_id, count in class_counts[split].items():
            print(f"Class {class_id}: {count} samples")