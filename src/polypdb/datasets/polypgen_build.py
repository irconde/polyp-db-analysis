from __future__ import annotations

"""Build manifest pack for the PolypGen dataset."""

from pathlib import Path
import subprocess

import pandas as pd

from ..manifests.core import ManifestRow
from ..manifests.io import write_pack


def build_polypgen_eval(pg_root: Path, out_dir: Path, track: str = "clean") -> None:
    """Build an evaluation manifest pack for PolypGen.

    Parameters
    ----------
    pg_root:
        Root directory of the canonical PolypGen dataset containing ``images`` and
        ``manifests`` subdirectories.
    out_dir:
        Directory where the manifest pack will be written.
    track:
        Dataset track name (e.g. ``"clean"`` or ``"raw"``).  This determines the
        ``store_id`` used in the manifest.
    """

    frames_path = pg_root / "manifests" / "frames.parquet"
    if not frames_path.exists():
        raise FileNotFoundError(f"Cannot locate frames metadata: {frames_path}")

    df = pd.read_parquet(frames_path)

    path_col = None
    for col in ["img_path", "frame_path", "path"]:
        if col in df.columns:
            path_col = col
            break
    if path_col is None:
        raise ValueError("frames.parquet missing frame path column")

    label_col = "has_polyp" if "has_polyp" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError("frames.parquet missing label column")

    rows: list[ManifestRow] = []
    for row in df.itertuples(index=False):
        frame_path = Path(getattr(row, path_col))
        if frame_path.is_absolute():
            rel_path = frame_path.relative_to(pg_root)
        else:
            rel_path = frame_path
        full_path = pg_root / rel_path
        if not full_path.exists():
            # pragma: no cover - allow missing files in source dataset
            continue
        rel = rel_path.as_posix()
        frame_id = rel.rsplit(".", 1)[0]
        label = int(getattr(row, label_col))
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="eval",
                case_id="",
                frame_id=frame_id,
                frame_path=rel,
                label=label,
                store_id=f"polypgen_{track}",
            )
        )

    command = f"polypdb polypgen build --root {pg_root} --out {out_dir}"
    if track:
        command += f" --track {track}"
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:  # pragma: no cover - fallback when git unavailable
        git_commit = "unknown"

    n_pos = sum(r.label == 1 for r in rows)
    n_neg = sum(r.label == 0 for r in rows)

    meta = {
        "name": f"polypgen-{track}-eval",
        "version": "0.0",
        "generator": {
            "repo": "polyp-db-analysis",
            "git_commit": git_commit,
            "command": command,
        },
        "policy": {"unit_of_split": "frame"},
        "roots": {f"polypgen_{track}": str(pg_root)},
        "assertions": {
            "class_presence": {"pos": ">0", "neg": ">0"},
            "label_counts": {"pos": n_pos, "neg": n_neg},
            "no_train_val_test_in_eval_pack": True,
        },
    }

    write_pack(out_dir, {"eval": rows}, meta)


__all__ = ["build_polypgen_eval"]
