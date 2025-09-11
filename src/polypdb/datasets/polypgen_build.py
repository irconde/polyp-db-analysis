from __future__ import annotations

"""Build manifest pack for the PolypGen dataset."""

from pathlib import Path
import subprocess

from ..manifests.core import ManifestRow
from ..manifests.io import write_pack

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def build_polypgen_eval(pg_root: Path, out_dir: Path, track: str = "clean") -> None:
    """Build an evaluation manifest pack for PolypGen.

    Parameters
    ----------
    pg_root:
        Root directory containing PolypGen frames.
    out_dir:
        Directory where the manifest pack will be written.
    track:
        Dataset track name (e.g. ``"clean"`` or ``"raw"``).  This determines the
        ``store_id`` used in the manifest.
    """

    rows: list[ManifestRow] = []
    for path in sorted(pg_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(pg_root).as_posix()
        frame_id = rel.rsplit(".", 1)[0]
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="eval",
                case_id="",
                frame_id=frame_id,
                frame_path=rel,
                label=1,
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
    }

    write_pack(out_dir, {"eval": rows}, meta)


__all__ = ["build_polypgen_eval"]
