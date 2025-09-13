"""Core manifest utilities.

This module defines data structures and helpers used when working with
manifest files.  Functions are intentionally deterministic so that the
same inputs always yield the same outputs across platforms and Python
versions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar, List
import hashlib
import hmac

T = TypeVar("T")


@dataclass
class ManifestRow:
    """A row in a manifest describing a single frame.

    Attributes mirror the fields commonly stored in manifest CSV files.
    Optional metadata defaults to sentinel values when unavailable.
    """

    dataset: str
    split: str
    case_id: str
    frame_id: str
    frame_path: str
    label: int
    store_id: str
    morphology: str = ""
    source_video: str = ""
    frame_index: int = -1
    width: int = -1
    height: int = -1
    variant: str = ""
    synthetic_case_id: str = ""
    source_case_id: str = ""
    chunk_start: int = -1
    chunk_end: int = -1
    split_unit: str = "case"


def hmac_rank(seed: int, text: str) -> int:
    """Return a 256-bit integer rank via HMAC-SHA256.

    The ``seed`` is encoded as an 8-byte big-endian key and the ``text`` is
    encoded as UTF-8.  The resulting digest is interpreted as a big-endian
    integer.
    """

    key = seed.to_bytes(8, "big", signed=False)
    digest = hmac.new(key, text.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest, "big")


def pick_k(
    items: Sequence[T], seed: int, k: int, key: Callable[[T], str]
) -> List[T]:
    """Select ``k`` items from ``items`` deterministically.

    Items are sorted by the HMAC rank of their key using ``seed``.  The
    ``key`` function should return a stable string representation for each
    item.  Sorting is deterministic and stable across Python versions.
    """

    ranked = sorted(items, key=lambda item: (hmac_rank(seed, key(item)), key(item)))
    return list(ranked[:k])


def sha256_file(path: Path) -> str:
    """Compute the SHA256 hex digest of a file.

    The file is read in binary mode in chunks to support large files.
    """

    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()
