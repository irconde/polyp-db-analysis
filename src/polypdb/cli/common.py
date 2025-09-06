"""Shared utilities for CLI sub‑commands."""

from __future__ import annotations

from pathlib import Path
from typing import Type, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str, model: Type[T] | None = None) -> T | dict:
    """Load configuration options from a YAML file.

    Parameters
    ----------
    path:
        Location of the YAML configuration file.
    model:
        Optional dataclass providing a ``from_dict`` classmethod used to
        validate and construct a typed configuration object.
    """

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if model is None:
        return data

    return model.from_dict(data, path=Path(path))


__all__ = ["load_config"]

