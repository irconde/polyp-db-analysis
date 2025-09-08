"""Implementation of the ``convert`` sub-command.

Converts raw datasets into the standardized layout without merging or
preparing YOLO datasets. Useful for running a single dataset conversion
using the same YAML config structure as the pipeline.
"""

from __future__ import annotations

import logging

from ..preprocess import CONVERTERS
from .common import load_config

logger = logging.getLogger(__name__)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "convert", help="Convert raw dataset(s) to the standard layout"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--dataset",
        choices=sorted(CONVERTERS.keys()),
        help="Convert only the specified dataset; defaults to all in config",
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
            # Maintain compatibility with configs that use 'metadata_file'
            if "metadata_file" in kwargs:
                kwargs["metadata_filename"] = kwargs.pop("metadata_file")
            converter = CONVERTERS[name](src, base_path, **kwargs)
            converter.convert()
        else:
            converter = CONVERTERS[name](src_cfg, base_path)
            converter.convert()
        logger.info("Finished converting %s", name)


__all__ = ["register"]

