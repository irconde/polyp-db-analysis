"""Shared utilities for CLI sub‑commands."""

from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    """Load configuration options from a YAML file."""

    with open(path, "r") as f:
        return yaml.safe_load(f)


__all__ = ["load_config"]

