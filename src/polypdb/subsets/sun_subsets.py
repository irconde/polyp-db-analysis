from __future__ import annotations

"""Utilities for building SUN training subsets.

This module constructs reduced training splits from a ``sun_full`` manifest
pack.  For each percentage and seed, cases are deterministically selected using
:func:`pick_k` and written to a new pack containing only ``train.csv``.  Metadata
for ``val`` and ``test`` splits is copied from the original pack without
duplicating their CSV files.
"""

from pathlib import Path
from datetime import datetime, timezone
import yaml

from ..manifests.core import pick_k, sha256_file
from ..manifests.io import read_csv, write_csv

CANONICAL_SUBSET_PERCENTAGES: tuple[int, ...] = (5, 10, 25, 50, 100)
CANONICAL_SUBSET_SEED = 13


def build_sun_train_subsets(
    sun_full_pack: Path,
    percents: list[int],
    seeds: list[int],
    out_dir: Path,
    *,
    percent_seed_map: dict[int, int] | None = None,
    generator_command: str | None = None,
) -> None:
    """Create training subsets of a ``sun_full`` manifest pack."""

    full_manifest_path = sun_full_pack / "manifest.yaml"
    full_manifest = yaml.safe_load(full_manifest_path.read_text(encoding="utf-8"))
    train_csv_full = sun_full_pack / "train.csv"
    train_rows = read_csv(train_csv_full)
    parent_train_hash = sha256_file(train_csv_full)

    cases_by_label: dict[int, set[str]] = {0: set(), 1: set()}
    for row in train_rows:
        cases_by_label[row.label].add(row.case_id)
    case_lists = {label: sorted(ids) for label, ids in cases_by_label.items()}

    if percent_seed_map:
        percent_seed_pairs = sorted(percent_seed_map.items())
    else:
        percent_seed_pairs = [(percent, seed) for percent in percents for seed in seeds]

    unique_seeds = {seed for _, seed in percent_seed_pairs}
    include_seed_in_name = len(unique_seeds) > 1

    if generator_command is None:
        percents_arg = " ".join(str(p) for p in sorted({p for p, _ in percent_seed_pairs}))
        seeds_arg = " ".join(str(s) for s in sorted(unique_seeds))
        default_command = (
            f"polypdb sun subsets --pack {sun_full_pack} --percents {percents_arg}"
            f" --seeds {seeds_arg} --out {out_dir}"
        )
    else:
        default_command = generator_command

    for percent, seed in percent_seed_pairs:
        selected_cases: set[str] = set()
        for label, case_ids in case_lists.items():
            k = round(len(case_ids) * percent / 100)
            if k:
                chosen = pick_k(case_ids, seed, k, key=lambda x: x)
                selected_cases.update(chosen)
        subset_rows = [r for r in train_rows if r.case_id in selected_cases]

        pack_suffix = f"_seed{seed}" if include_seed_in_name else ""
        pack_name = f"sun_p{percent:02d}{pack_suffix}"
        pack_dir = out_dir / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)

        train_csv = pack_dir / "train.csv"
        write_csv(subset_rows, train_csv)
        train_hash = sha256_file(train_csv)

        counts_train = {
            "frames": len(subset_rows),
            "pos": sum(1 for r in subset_rows if r.label == 1),
            "neg": sum(1 for r in subset_rows if r.label == 0),
            "pos_cases": len({r.case_id for r in subset_rows if r.label == 1}),
            "neg_cases": len({r.case_id for r in subset_rows if r.label == 0}),
        }

        manifest = dict(full_manifest)
        manifest["name"] = pack_name
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

        hashes = dict(full_manifest.get("hashes", {}))
        hashes["train"] = train_hash
        manifest["hashes"] = hashes

        counts = dict(full_manifest.get("counts", {}))
        counts["train"] = counts_train
        manifest["counts"] = counts

        # parent pack provenance
        parent_pack = {
            "name": "sun_full",
            "hashes": {
                "val": hashes.get("val", ""),
                "test": hashes.get("test", ""),
                "train": parent_train_hash,
            },
            "path": str(sun_full_pack),
        }
        manifest["parent_pack"] = parent_pack

        # subset metadata
        subset_meta = {
            "percent": percent,
            "seed": seed,
            "source_pool": "train",
        }
        if percent_seed_map:
            subset_meta["canonical"] = True
        manifest["subset"] = subset_meta

        # additional assertions
        assertions = dict(full_manifest.get("assertions", {}))
        assertions.update(
            {
                "subset_train_from_parent_train": True,
                "subset_train_pos_eq_neg": counts_train["pos"]
                == counts_train["neg"],
                "val_test_hash_match_parent": True,
            }
        )
        manifest["assertions"] = assertions

        # selection scopes
        policy = dict(full_manifest.get("policy", {}))
        policy["selection_scope"] = "case"
        manifest["policy"] = policy

        frame_sampling = dict(full_manifest.get("frame_sampling", {}))
        frame_sampling["selection_scope"] = "frame"
        manifest["frame_sampling"] = frame_sampling

        # update splits metadata
        splits_meta = dict(full_manifest.get("splits", {}))
        splits_meta["train"] = {
            "pos_cases": sorted({r.case_id for r in subset_rows if r.label == 1}),
            "neg_cases": sorted({r.case_id for r in subset_rows if r.label == 0}),
        }
        manifest["splits"] = splits_meta

        # update summary table
        summary = list(full_manifest.get("summary_by_split", []))
        pos_case_ids = splits_meta["train"]["pos_cases"]
        neg_case_ids = splits_meta["train"]["neg_cases"]
        new_train_summary = {
            "split": "train",
            "pos_cases": len(pos_case_ids),
            "neg_cases": len(neg_case_ids),
            "frames": len(subset_rows),
            "pos_frames": counts_train["pos"],
            "neg_frames": counts_train["neg"],
        }
        summary = [s for s in summary if s.get("split") != "train"]
        summary.append(new_train_summary)
        manifest["summary_by_split"] = summary

        total_cases = sum(s.get("pos_cases", 0) + s.get("neg_cases", 0) for s in summary)
        manifest["case_lengths"] = {33: total_cases}

        generator = dict(full_manifest.get("generator", {}))
        generator["command"] = default_command
        generator["seed"] = seed
        if percent_seed_map:
            generator["percent_seed_map"] = percent_seed_map
        manifest["generator"] = generator

        with (pack_dir / "manifest.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)


def build_sun_canonical_subsets(
    sun_full_pack: Path,
    out_dir: Path,
    *,
    seed: int = CANONICAL_SUBSET_SEED,
    percents: tuple[int, ...] = CANONICAL_SUBSET_PERCENTAGES,
) -> None:
    """Build the canonical SUN train subset packs.

    The canonical configuration uses a single seed per percentage to ensure
    reproducibility across releases.  By default the seed ``13`` is applied to
    the 5%, 10%, 25%, 50%, and 100% subsets of the SUN training split.
    """

    ordered_percents = tuple(sorted(percents))
    percent_seed_map = {percent: seed for percent in ordered_percents}
    percents_arg = " ".join(str(p) for p in ordered_percents)
    generator_command = (
        f"polypdb sun subsets-canonical --pack {sun_full_pack}"
        f" --percents {percents_arg} --seed {seed} --out {out_dir}"
    )
    build_sun_train_subsets(
        sun_full_pack,
        percents=list(ordered_percents),
        seeds=[seed],
        out_dir=out_dir,
        percent_seed_map=percent_seed_map,
        generator_command=generator_command,
    )


__all__ = ["build_sun_train_subsets", "build_sun_canonical_subsets", "CANONICAL_SUBSET_PERCENTAGES", "CANONICAL_SUBSET_SEED"]
