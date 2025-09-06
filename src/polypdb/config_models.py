"""Data models for configuration files.

Defines typed dataclasses representing the expected structure of
configuration files used throughout the project.  Using these models makes
interfaces explicit and provides early validation of configuration files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MergeConfig:
    """Configuration options for :func:`merge_datasets`."""

    base_path: Path
    datasets: List[str]
    num_elements: Optional[int] = None
    plot_images: bool = False
    negative_sources: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, *, path: Path | None = None) -> "MergeConfig":
        missing = {"base_path", "datasets"} - data.keys()
        if missing:
            raise ValueError(
                "Missing required config keys: " + ", ".join(sorted(missing))
            )
        datasets = data["datasets"]
        if not isinstance(datasets, list):
            raise TypeError("'datasets' must be a list")
        negative_sources = data.get("negative_sources") or {}
        if negative_sources and not isinstance(negative_sources, dict):
            raise TypeError("'negative_sources' must be a mapping if provided")
        return cls(
            base_path=Path(data["base_path"]),
            datasets=list(datasets),
            num_elements=data.get("num_elements"),
            plot_images=bool(data.get("plot_images", False)),
            negative_sources=negative_sources,
        )


@dataclass
class SplittingOptions:
    """Options controlling how a dataset is split into train/val/test sets."""

    train_ratio: float
    validation_ratio: float
    test_ratio: float
    split_type: str
    split_level: str

    @classmethod
    def from_dict(cls, data: dict) -> "SplittingOptions":
        required = {
            "train_ratio",
            "validation_ratio",
            "test_ratio",
            "split_type",
            "split_level",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(
                "Missing required splitting_options keys: "
                + ", ".join(sorted(missing))
            )
        return cls(
            train_ratio=data["train_ratio"],
            validation_ratio=data["validation_ratio"],
            test_ratio=data["test_ratio"],
            split_type=data["split_type"],
            split_level=data["split_level"],
        )


@dataclass
class NegativeSampling:
    """Configuration for negative sample handling."""

    ratio: float
    distribution: str

    @classmethod
    def from_dict(cls, data: dict) -> "NegativeSampling":
        required = {"ratio", "distribution"}
        missing = required - data.keys()
        if missing:
            raise ValueError(
                "Missing required negative_sampling keys: "
                + ", ".join(sorted(missing))
            )
        return cls(ratio=data["ratio"], distribution=data["distribution"])


@dataclass
class PrepareConfig:
    """Configuration options for dataset preparation."""

    src_path: Path
    classification_type: str
    splitting_options: SplittingOptions
    negative_sampling: NegativeSampling
    config_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict, *, path: Path | None = None) -> "PrepareConfig":
        required = {
            "src_path",
            "classification_type",
            "splitting_options",
            "negative_sampling",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(
                "Missing required config keys: " + ", ".join(sorted(missing))
            )
        return cls(
            src_path=Path(data["src_path"]),
            classification_type=data["classification_type"],
            splitting_options=SplittingOptions.from_dict(data["splitting_options"]),
            negative_sampling=NegativeSampling.from_dict(data["negative_sampling"]),
            config_path=path,
        )


__all__ = [
    "MergeConfig",
    "PrepareConfig",
    "SplittingOptions",
    "NegativeSampling",
]
