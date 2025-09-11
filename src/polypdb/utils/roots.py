from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

import yaml

from ..manifests.core import ManifestRow


def load_roots(
    roots_json: Path | None,
    manifest_yaml: Path | None,
    default_root: Path | None,
) -> Dict[str, str]:
    """Load store roots from multiple sources.

    Roots are resolved in descending priority.  Values from ``roots_json``
    override those from ``manifest_yaml`` which in turn override the
    ``default_root``.  ``default_root`` is stored under the special key ``"*"``
    and is used when resolving paths for unknown ``store_id`` values.
    """

    roots: Dict[str, str] = {}

    if default_root is not None:
        roots["*"] = str(default_root)

    if manifest_yaml is not None and manifest_yaml.exists():
        manifest = yaml.safe_load(manifest_yaml.read_text(encoding="utf-8"))
        roots.update(manifest.get("roots", {}))

    if roots_json is not None:
        cli_roots = json.loads(roots_json.read_text(encoding="utf-8"))
        roots.update(cli_roots)

    return roots


def resolve_path(row: ManifestRow, roots: Dict[str, str]) -> Path:
    """Resolve the file path for ``row`` using ``roots``.

    ``roots`` is expected to map ``store_id`` values to filesystem paths.  The
    special key ``"*"`` may hold a default root used when ``row.store_id`` is
    missing from the mapping.
    """

    root = roots.get(row.store_id, roots.get("*"))
    if root is None:
        raise KeyError(f"missing root for store_id {row.store_id}")
    return Path(root) / row.frame_path
