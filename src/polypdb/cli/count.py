"""Implementation of the ``count`` sub‑command."""

from __future__ import annotations

import logging

from ..count_samples import count_samples

logger = logging.getLogger(__name__)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "count", help="Count class samples in YOLO dataset"
    )
    parser.add_argument("root_folder", help="Root folder of the dataset")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    counts = count_samples(args.root_folder)
    for split in ["test", "train", "val"]:
        logger.info("%s split:", split.capitalize())
        for class_id, cnt in counts[split].items():
            logger.info("Class %s: %s samples", class_id, cnt)


__all__ = ["register"]

