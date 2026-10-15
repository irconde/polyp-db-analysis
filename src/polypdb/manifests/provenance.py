"""Provenance helpers recorded in every manifest pack."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import platform
import subprocess

from .core import sha256_file

_PACKAGE_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_PACKAGE_DIR), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # pragma: no cover - git unavailable / not a checkout
        return None


@lru_cache(maxsize=1)
def tool_provenance() -> Dict[str, Any]:
    """Return the commit and dirty state of the *tool* repository plus versions.

    The commit is resolved from the location of this package, not from the
    current working directory, so it is correct wherever the CLI is launched.
    """

    import numpy
    import pandas

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit.strip() if commit else "unknown",
        "git_dirty": (bool(status.strip()) if status is not None else None),
        "environment": {
            "python": platform.python_version(),
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
        },
    }


def input_hashes(roots: Dict[str, str]) -> Dict[str, Dict[str, str | None]]:
    """SHA-256 of ``frames.parquet`` and ``cases.parquet`` for each root."""

    out: Dict[str, Dict[str, str | None]] = {}
    for store_id, root in sorted((roots or {}).items()):
        manifests = Path(root) / "manifests"
        entry: Dict[str, str | None] = {}
        for name in ("frames", "cases"):
            path = manifests / f"{name}.parquet"
            entry[f"{name}_parquet_sha256"] = (
                sha256_file(path) if path.exists() else None
            )
        out[store_id] = entry
    return out


def parent_pack_info(pack_dir: Path) -> Dict[str, Any]:
    """Identify a parent pack by manifest hash and CSV hashes."""

    import yaml

    manifest_path = pack_dir / "manifest.yaml"
    info: Dict[str, Any] = {"name": pack_dir.name, "path": str(pack_dir)}
    if manifest_path.exists():
        info["manifest_sha256"] = sha256_file(manifest_path)
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        info["name"] = manifest.get("name", pack_dir.name)
    info["hashes"] = {
        p.stem: sha256_file(p) for p in sorted(pack_dir.glob("*.csv"))
    }
    return info


def stamp_manifest(manifest: Dict[str, Any]) -> None:
    """Record tool commit, dirty state, environment and input hashes in place."""

    prov = tool_provenance()
    generator = dict(manifest.get("generator") or {})
    generator["git_commit"] = prov["git_commit"]
    generator["git_dirty"] = prov["git_dirty"]
    generator["environment"] = dict(prov["environment"])
    manifest["generator"] = generator
    roots = manifest.get("roots")
    if roots:
        manifest["inputs"] = input_hashes(roots)
