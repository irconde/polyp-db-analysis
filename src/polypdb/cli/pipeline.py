"""Implementation of the ``pipeline`` sub‑command."""

from __future__ import annotations

import logging
import os

from ..preprocess import CONVERTERS
from ..ultralytics import prepare_classification_dataset, prepare_dataset
from .common import load_config

logger = logging.getLogger(__name__)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "pipeline", help="Run full preprocessing pipeline"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--dataset",
        choices=sorted(CONVERTERS.keys()),
        help="Process only the specified dataset instead of merging all",
    )
    parser.set_defaults(func=_run)


def _run(args) -> None:
    cfg = load_config(args.config)
    if args.dataset and args.dataset not in cfg.get("raw_sources", {}):
        raise ValueError(f"Dataset '{args.dataset}' not defined in configuration")
    base_path = cfg["base_path"]
    datasets = [args.dataset] if args.dataset else cfg["datasets"]

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

    from . import merge_datasets

    logger.info("Merging datasets")

    negative_sources = cfg.get("negative_sources")
    if args.dataset and negative_sources:
        ns_val = negative_sources.get(args.dataset)
        negative_sources = {args.dataset: ns_val} if ns_val else None

    merge_datasets(
        base_path=base_path,
        datasets=datasets,
        num_elements=cfg.get("num_elements"),
        negative_sources=negative_sources,
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


__all__ = ["register"]

