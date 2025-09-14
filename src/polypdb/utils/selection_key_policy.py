from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_selection_key_policy(manifest_yaml: Path | None) -> Optional[Dict[str, Any]]:
    """Load ``selection_key_policy`` from ``manifest_yaml`` if available.

    Returns ``None`` when ``manifest_yaml`` is ``None`` or does not exist, or
    when the file does not define a ``selection_key_policy`` block.
    """

    if manifest_yaml is None or not manifest_yaml.exists():
        return None
    manifest = yaml.safe_load(manifest_yaml.read_text(encoding="utf-8"))
    return manifest.get("selection_key_policy")
