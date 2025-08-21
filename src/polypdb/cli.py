import argparse
import os
import yaml
import logging

from .merge import merge_datasets
from .preprocess import CONVERTERS
from .ultralytics import prepare_dataset, prepare_classification_dataset
from .count_samples import count_samples


logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set the logging level (default: INFO)",
    )
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

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
            logger.info("%s split:", split.capitalize())
            for class_id, cnt in counts[split].items():
                logger.info("Class %s: %s samples", class_id, cnt)
    elif args.command == "pipeline":
        cfg = load_config(args.config)
        base_path = cfg["base_path"]
        datasets = cfg["datasets"]
        for idx, name in enumerate(datasets, start=1):
            logger.info("(%d/%d) Converting dataset %s", idx, len(datasets), name)
            src_cfg = cfg["raw_sources"][name]
            if isinstance(src_cfg, dict):
                src = src_cfg.get("path")
                metadata_file = src_cfg.get("metadata_file")
                if metadata_file:
                    CONVERTERS[name](src, base_path, metadata_filename=metadata_file)
                else:
                    CONVERTERS[name](src, base_path)
            else:
                CONVERTERS[name](src_cfg, base_path)
            logger.info("Finished converting %s", name)
        logger.info("Merging datasets")
        merge_datasets(base_path=base_path, datasets=datasets, num_elements=cfg.get("num_elements"))
        if cfg.get("prepare"):
            prepare_cfg = cfg.get("prepare_config")
            if not prepare_cfg:
                raise ValueError("Missing 'prepare_config' in pipeline configuration")
            prepare_cfg = os.path.expanduser(prepare_cfg)
            if not os.path.isabs(prepare_cfg):
                prepare_cfg = os.path.join(base_path, prepare_cfg)
            if cfg.get("classification"):
                logger.info("Preparing classification dataset")
                prepare_classification_dataset(prepare_cfg)
            else:
                logger.info("Preparing detection dataset")
                prepare_dataset(prepare_cfg)


if __name__ == "__main__":
    main()
