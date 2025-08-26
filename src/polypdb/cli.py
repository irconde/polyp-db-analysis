import argparse
import os
import yaml
import logging

from .merge import merge_datasets, dataset_statistics_from_canonical
from .preprocess import CONVERTERS
from .ultralytics import prepare_dataset, prepare_classification_dataset
from .count_samples import count_samples
from .utils import crop_and_mask_corner_artifacts, create_text_masks


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

    stats_parser = subparsers.add_parser(
        "stats", help="Compute dataset statistics from a canonical folder"
    )
    stats_parser.add_argument(
        "canonical_dir", help="Path to the canonical dataset directory"
    )

    mask_parser = subparsers.add_parser(
        "mask-corners", help="Mask corner artifacts then crop images"
    )
    mask_parser.add_argument("src_dir", help="Directory containing input images")
    mask_parser.add_argument(
        "dest_dir", help="Directory to save processed images"
    )
    mask_parser.add_argument(
        "--padding-color",
        default="15,15,15",
        help="RGB padding colour to strip after masking (default: 15,15,15)",
    )
    mask_parser.add_argument(
        "--padding-threshold",
        type=int,
        default=20,
        help="Colour deviation allowed when removing padding (default: 20)",
    )
    mask_parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Colour deviation threshold for corner artefact detection",
    )
    mask_parser.add_argument(
        "--max-auto-fraction",
        type=float,
        default=0.15,
        help=(
            "Maximum fraction of the image dimension that auto detection may mask "
            "when removing corner artefacts (default: 0.15)"
        ),
    )

    text_mask_parser = subparsers.add_parser(
        "text-masks", help="Detect text and create mask images"
    )
    text_mask_parser.add_argument("src_dir", help="Directory containing input images")
    text_mask_parser.add_argument(
        "dest_dir", help="Directory to save generated mask images"
    )
    text_mask_parser.add_argument(
        "--inpaint-dir",
        help="Directory to save images with text removed",
    )
    text_mask_parser.add_argument(
        "--expand",
        type=int,
        default=5,
        help="Pixels to expand each detected text mask (default: 5)",
    )
    text_mask_parser.add_argument(
        "--debug-dir",
        help="Directory to save debug images with mask contours",
    )

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
    elif args.command == "stats":
        dataset_statistics_from_canonical(args.canonical_dir)
    elif args.command == "mask-corners":
        padding_color = tuple(int(c) for c in args.padding_color.split(","))
        crop_and_mask_corner_artifacts(
            args.src_dir,
            args.dest_dir,
            padding_color=padding_color,
            padding_threshold=args.padding_threshold,
            threshold=args.threshold,
            max_auto_fraction=args.max_auto_fraction,
        )
    elif args.command == "text-masks":
        create_text_masks(
            args.src_dir,
            args.dest_dir,
            expand=args.expand,
            inpaint_dir=args.inpaint_dir,
            debug_dir=args.debug_dir,
        )
    elif args.command == "pipeline":
        cfg = load_config(args.config)
        base_path = cfg["base_path"]
        datasets = cfg["datasets"]
        for idx, name in enumerate(datasets, start=1):
            logger.info("(%d/%d) Converting dataset %s", idx, len(datasets), name)
            src_cfg = cfg["raw_sources"][name]
            if isinstance(src_cfg, dict):
                src = src_cfg.get("path")
                kwargs = {k: v for k, v in src_cfg.items() if k != "path"}
                if "metadata_file" in kwargs:
                    kwargs["metadata_filename"] = kwargs.pop("metadata_file")
                CONVERTERS[name](src, base_path, **kwargs)
            else:
                CONVERTERS[name](src_cfg, base_path)
            logger.info("Finished converting %s", name)
        logger.info("Merging datasets")
        merge_datasets(
            base_path=base_path,
            datasets=datasets,
            num_elements=cfg.get("num_elements"),
            negative_sources=cfg.get("negative_sources"),
        )
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
