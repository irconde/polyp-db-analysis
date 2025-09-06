from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DatasetConverter:
    """Base class for converting datasets to the canonical layout.

    Sub-classes should implement the :meth:`load_metadata`,
    :meth:`normalize_columns`, :meth:`compute_dimensions` and
    :meth:`write_outputs` hooks.  The default :meth:`convert` method calls these
    hooks in order and handles passing the intermediate :class:`pandas.DataFrame`
    between them.
    """

    def __init__(self, src_dir: Path | str, dest_dir: Path | str, **kwargs: Any) -> None:
        self.src_dir = Path(src_dir)
        self.dest_dir = Path(dest_dir)
        self.kwargs = kwargs

    # Hook methods -----------------------------------------------------
    def load_metadata(self) -> pd.DataFrame:  # pragma: no cover - interface
        """Return a DataFrame with the raw metadata for the dataset."""
        raise NotImplementedError

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Return ``df`` with dataset specific normalisation applied."""
        return df

    def compute_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Return ``df`` with additional columns derived from image data."""
        return df

    def write_outputs(self, df: pd.DataFrame) -> None:  # pragma: no cover - interface
        """Persist the converted dataset to ``dest_dir``."""
        raise NotImplementedError

    # Public API -------------------------------------------------------
    def convert(self) -> None:
        """Execute the conversion pipeline."""
        df = self.load_metadata()
        df = self.normalize_columns(df)
        df = self.compute_dimensions(df)
        self.write_outputs(df)
