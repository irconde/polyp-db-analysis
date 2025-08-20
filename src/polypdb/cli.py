import argparse
import os
import yaml

from .merge import merge_datasets
from .preprocess import CONVERTERS
from .ultralytics import prepare_dataset, prepare_classification_dataset
from .count_samples import count_samples


def load_config(path: str) -> dict:
    """Load configuration options from a YAML file.

    Parameters
    ----------
    path : str
        Location of the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run the command-line interface for dataset utilities."""
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

    pipeline_parser = subparsers.add_parser("pipeline", help="Run full preprocessing pipeline")
    pipeline_parser.add_argument("config", help="Path to YAML configuration file")

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
    elif args.command == "pipeline":
        cfg = load_config(args.config)
        base_path = cfg["base_path"]
        for name in cfg["datasets"]:
            src = cfg["raw_sources"][name]
            CONVERTERS[name](src, base_path)
        merge_datasets(base_path=base_path, datasets=cfg["datasets"], num_elements=cfg.get("num_elements"))
        if cfg.get("prepare"):
            prepare_cfg = cfg.get("prepare_config", os.path.join(base_path, "ultralytics_config.yaml"))
            if cfg.get("classification"):
                prepare_classification_dataset(prepare_cfg)
            else:
                prepare_dataset(prepare_cfg)


if __name__ == "__main__":
    main()
