import argparse
import yaml

from .merge import merge_datasets
from .ultralytics import prepare_dataset, prepare_classification_dataset
from .count_samples import count_samples


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polyp dataset processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge datasets into a unified dataset")
    merge_parser.add_argument("config", help="Path to YAML configuration file")

    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare dataset for Ultralytics YOLO")
    prepare_parser.add_argument("config", help="Path to YAML configuration file")

    prepare_cls_parser = subparsers.add_parser(
        "prepare-classification", help="Prepare dataset for image classification"
    )
    prepare_cls_parser.add_argument("config", help="Path to YAML configuration file")

    count_parser = subparsers.add_parser("count", help="Count class samples in YOLO dataset")
    count_parser.add_argument("root_folder", help="Root folder of the dataset")

    args = parser.parse_args()

    if args.command == "merge":
        cfg = load_config(args.config)
        merge_datasets(**cfg)
    elif args.command == "prepare":
        prepare_dataset(args.config)
    elif args.command == "prepare-classification":
        prepare_classification_dataset(args.config)
    elif args.command == "count":
        counts = count_samples(args.root_folder)
        for split in ["test", "train", "val"]:
            print(f"\n{split.capitalize()} split:")
            for class_id, cnt in counts[split].items():
                print(f"Class {class_id}: {cnt} samples")


if __name__ == "__main__":
    main()
