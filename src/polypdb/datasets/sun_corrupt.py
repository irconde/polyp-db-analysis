from __future__ import annotations

"""Build a corrupted SUN manifest pack.

This utility reads an existing ``sun_full`` pack and writes a new pack where the
training split rows are marked with a ``variant`` value of ``"corrupt"``.
The output pack retains the original validation and test splits.
"""

from dataclasses import replace
from pathlib import Path
import subprocess

from ..manifests.io import read_csv, write_pack


def build_sun_corrupt(sun_full_pack: Path, out_dir: Path, variant: str = "corrupt") -> None:
    """Create a corrupted SUN pack from ``sun_full_pack``.

    Parameters
    ----------
    sun_full_pack:
        Path to an existing ``sun_full`` pack directory.
    out_dir:
        Directory where the corrupted pack will be written.
    variant:
        Value assigned to the ``variant`` field for training rows.  Defaults to
        ``"corrupt"``.
    """

    splits = {}
    for split in ("train", "val", "test"):
        csv_path = sun_full_pack / f"{split}.csv"
        if not csv_path.exists():
            continue
        rows = read_csv(csv_path)
        if split == "train":
            rows = [replace(r, variant=variant) for r in rows]
        splits[split] = rows

    command = f"polypdb sun corrupt --pack {sun_full_pack} --out {out_dir}"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git not available
        git_commit = "unknown"

    meta = {
        "name": f"sun_{variant}",
        "version": "0.0",
        "generator": {
            "repo": "polyp-db-analysis",
            "git_commit": git_commit,
            "command": command,
        },
        "policy": {"unit_of_split": "frame", "variant": variant},
        "roots": {},
    }

    write_pack(out_dir, splits, meta)


__all__ = ["build_sun_corrupt"]
