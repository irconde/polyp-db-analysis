from __future__ import annotations

"""Build manifest pack for the PolypGen dataset."""

from pathlib import Path
import subprocess
import logging

import pandas as pd

from ..manifests.core import ManifestRow
from ..manifests.io import write_pack


logger = logging.getLogger(__name__)


def build_polypgen_eval(
    pg_root: Path,
    out_dir: Path,
    track: str = "clean",
    *,
    include_negative_sequences: bool = False,
) -> None:
    """Build a test manifest pack for PolypGen.

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
    include_negative_sequences:
        When ``True`` retain negative frames that originate from sequences.
        The default ``False`` matches the historical ``polypgen_clean_test``
        pack which excluded them.
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

    if "global_frame_id" not in df.columns:
        raise ValueError("frames.parquet missing global_frame_id column")
    if "center_id" not in df.columns:
        raise ValueError("frames.parquet missing center_id column")

    width_col = "img_width" if "img_width" in df.columns else None
    height_col = "img_height" if "img_height" in df.columns else None

    if track == "clean" and not include_negative_sequences:
        seq_mask = pd.Series(False, index=df.index)
        if "origin" in df.columns:
            origin_series = df["origin"].astype("string")
            origin_lower = origin_series.str.lower()
            seq_from_origin = origin_lower == "neg_seq"
            seq_mask = seq_mask | seq_from_origin.fillna(False)
            if "sequence_id" in df.columns:
                seq_mask = seq_mask | (origin_series.isna() & df["sequence_id"].notna())
        elif "sequence_id" in df.columns:
            seq_mask = seq_mask | df["sequence_id"].notna()
        drop_mask = (df[label_col] == 0) & seq_mask
        drop_count = int(drop_mask.sum())
        if drop_count:
            logger.info(
                "Filtering %d PolypGen clean-track negatives originating from sequences",
                drop_count,
            )
            df = df.loc[~drop_mask].copy()

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
        frame_id = str(getattr(row, "global_frame_id"))
        label = int(getattr(row, label_col))
        case_id = str(getattr(row, "center_id"))
        width = (
            int(getattr(row, width_col))
            if width_col and pd.notna(getattr(row, width_col))
            else -1
        )
        height = (
            int(getattr(row, height_col))
            if height_col and pd.notna(getattr(row, height_col))
            else -1
        )
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="test",
                case_id=case_id,
                frame_id=frame_id,
                frame_path=rel,
                label=label,
                store_id=f"polypgen_{track}",
                width=width,
                height=height,
            )
        )

    command = f"polypdb polypgen build --root {pg_root} --out {out_dir}"
    if track:
        command += f" --track {track}"
    if include_negative_sequences:
        command += " --include-negative-sequences"
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:  # pragma: no cover - fallback when git unavailable
        git_commit = "unknown"

    n_pos = sum(r.label == 1 for r in rows)
    n_neg = sum(r.label == 0 for r in rows)

    variant_suffix = "-extended" if include_negative_sequences else ""
    meta = {
        "name": f"polypgen-{track}-test{variant_suffix}",
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
            "no_train_val_test_in_test_pack": True,
        },
    }

    write_pack(out_dir, {"test": rows}, meta)


__all__ = ["build_polypgen_eval"]
