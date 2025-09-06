"""Implementation of the ``stats-manifests`` sub‑command."""

from __future__ import annotations

import json
import logging

from . import dataset_statistics_from_parquets

logger = logging.getLogger(__name__)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "stats-manifests",
        help="Compute dataset statistics from manifest parquet files",
    )
    parser.add_argument("--frames", required=True, help="Path to frames.parquet")
    parser.add_argument("--cases", help="Optional path to cases.parquet")
    parser.add_argument("--json-out", help="Optional path to write statistics as JSON")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    stats = dataset_statistics_from_parquets(args.frames, args.cases)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Wrote statistics to %s", args.json_out)
    else:
        logger.info("Original total frames: %s", stats.get("original_total_samples"))
        logger.info("Total frames (after dedup): %s", stats.get("total_samples"))
        logger.info("Total cases: %s", stats.get("total_cases"))
        logger.info("Case distribution: %s", stats.get("case_distribution"))
        logger.info(
            "Frames per case summary: %s", stats.get("frames_per_case_summary")
        )
        logger.info(
            "Morphology distribution: %s", stats.get("morphology_distribution")
        )
        logger.info(
            "Dataset case distribution: %s", stats.get("dataset_case_distribution")
        )
        logger.info(
            "Dataset frame distribution: %s", stats.get("dataset_frame_distribution")
        )


__all__ = ["register"]

