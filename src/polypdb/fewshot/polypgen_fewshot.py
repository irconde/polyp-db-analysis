from __future__ import annotations

"""Utilities for creating few-shot PolypGen training splits."""

from dataclasses import replace
from pathlib import Path
from math import ceil, floor
import subprocess
import yaml

from ..manifests.core import ManifestRow, pick_k
from ..manifests.io import read_csv, write_pack


def build_polypgen_fewshot(eval_pack: Path, sizes: list[int], seeds: list[int], out_dir: Path) -> None:
    """Build few-shot training/eval splits from a PolypGen evaluation pack.

    Parameters
    ----------
    eval_pack:
        Path to a manifest pack containing ``eval.csv`` with both positive and
        negative frames.
    sizes:
        Desired training set sizes.  For each size ``S`` and ``seed`` we select
        ``ceil(S/2)`` positive and ``floor(S/2)`` negative frames.
    seeds:
        Seeds used for deterministic selection.
    out_dir:
        Directory where the generated packs will be written.  Packs are named
        ``polypgen_fewshot_s{S}_seed{seed}``.
    """

    eval_rows = read_csv(eval_pack / "eval.csv")

    # Load manifest for roots if available
    roots = {}
    manifest_path = eval_pack / "manifest.yaml"
    if manifest_path.exists():
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            roots = manifest.get("roots", {})
        except Exception:
            roots = {}

    pos_rows = [r for r in eval_rows if r.label == 1]
    neg_rows = [r for r in eval_rows if r.label == 0]

    for size in sizes:
        for seed in seeds:
            k_pos = min(ceil(size / 2), len(pos_rows))
            k_neg = min(floor(size / 2), len(neg_rows))

            pos_selected = pick_k(pos_rows, seed, k_pos, key=lambda r: r.frame_id)
            neg_selected = pick_k(neg_rows, seed, k_neg, key=lambda r: r.frame_id)

            train_rows = [replace(r, split="train") for r in pos_selected + neg_selected]
            selected_ids = {r.frame_id for r in train_rows}
            eval_rows_out = [
                replace(r, split="eval") for r in eval_rows if r.frame_id not in selected_ids
            ]

            pack_dir = out_dir / f"polypgen_fewshot_s{size}_seed{seed}"

            meta = {
                "name": f"polypgen_fewshot_s{size}_seed{seed}",
                "version": "0.0",
                "policy": {
                    "unit_of_split": "frame",
                    "selection": f"hmac_sha256(frame_id,{seed})",
                },
                "roots": roots,
            }

            command = (
                f"polypdb polypgen fewshot --pack {eval_pack} --sizes {size} "
                f"--seeds {seed} --out {out_dir}"
            )
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
            except Exception:  # pragma: no cover
                git_commit = "unknown"
            meta["generator"] = {
                "repo": "polyp-db-analysis",
                "git_commit": git_commit,
                "command": command,
                "seed": seed,
            }

            write_pack(pack_dir, {"train": train_rows, "eval": eval_rows_out}, meta)


__all__ = ["build_polypgen_fewshot"]
