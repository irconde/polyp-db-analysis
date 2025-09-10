from __future__ import annotations

"""Utilities for preparing SUN dataset for classification."""

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
from typing import List, Dict

import pandas as pd


CHUNK_SIZE = 33
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PARIS_GROUPS: Dict[str, set[str]] = {
    "polypoid": {"is", "isp", "ip"},
    "flat": {"iia", "iib"},
    "depressed": {"iic", "iii"},
}


def _paris_to_group(paris: str | None) -> str | None:
    if paris is None:
        return None
    p = paris.lower()
    for group, members in PARIS_GROUPS.items():
        if p in members:
            return group
    return None


@dataclass
class _Case:
    """Helper structure holding frames for a single case."""

    case_id: str
    frames: pd.DataFrame
    has_polyp: bool
    new_id: str | None = None
    morphology: str | None = None


def _load_metadata(src: Path) -> pd.DataFrame:
    """Load joined frame metadata from canonical directory."""

    frames_path = src / "manifests" / "frames.parquet"
    cases_path = src / "manifests" / "cases.parquet"

    frames_cols = ["global_frame_id", "global_case_id", "has_polyp", "img_path"]
    frames_df = pd.read_parquet(frames_path, columns=frames_cols)

    if cases_path.exists():
        cases_df = pd.read_parquet(cases_path)
        cols = ["global_case_id"]
        if "global_lesion_id" in cases_df.columns:
            cols.append("global_lesion_id")
        if "paris_class" in cases_df.columns:
            cols.append("paris_class")
        frames_df = frames_df.merge(cases_df[cols], on="global_case_id", how="left")
    return frames_df


def _build_positive_cases(frames_df: pd.DataFrame, seed: int) -> List[_Case]:
    cases = []
    grouped = frames_df[frames_df["has_polyp"] == True].groupby("global_case_id")  # noqa: E712
    for case_id, group in grouped:
        group_sorted = group.sort_values("global_frame_id")
        if len(group_sorted) > CHUNK_SIZE:
            sampled = group_sorted.sample(n=CHUNK_SIZE, random_state=seed, replace=False)
            sampled = sampled.sort_values("global_frame_id")
        else:
            sampled = group_sorted
        paris = None
        if "paris_class" in group.columns:
            paris = group["paris_class"].iloc[0]
        morphology = _paris_to_group(paris)
        cases.append(
            _Case(case_id=str(case_id), frames=sampled, has_polyp=True, morphology=morphology)
        )
    return cases


def _build_negative_cases(frames_df: pd.DataFrame) -> List[_Case]:
    cases: List[_Case] = []
    grouped = frames_df[frames_df["has_polyp"] == False].groupby("global_case_id")  # noqa: E712
    for case_id, group in grouped:
        group_sorted = group.sort_values("global_frame_id")
        chunks = [group_sorted.iloc[i : i + CHUNK_SIZE] for i in range(0, len(group_sorted), CHUNK_SIZE)]
        for idx, chunk in enumerate(chunks):
            if len(chunk) < CHUNK_SIZE:
                continue
            synthetic_id = f"{case_id}_chunk{idx:03d}"
            cases.append(_Case(case_id=synthetic_id, frames=chunk, has_polyp=False))
    return cases


def _split_cases(cases: List[_Case]) -> tuple[List[_Case], List[_Case], List[_Case]]:
    n = len(cases)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    train = cases[:n_train]
    val = cases[n_train : n_train + n_val]
    test = cases[n_train + n_val :]
    return train, val, test


def _copy_case(case: _Case, src: Path, dest: Path) -> None:
    label = "polyp" if case.has_polyp else "non-polyp"
    out_dir = dest / label
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_sorted = case.frames.sort_values("global_frame_id")
    for idx, row in enumerate(frames_sorted.itertuples(index=False)):
        img_rel = getattr(row, "img_path")
        src_path = src / img_rel
        ext = Path(img_rel).suffix
        filename = f"{case.new_id}_f{idx:03d}{ext}"
        shutil.copy(src_path, out_dir / filename)


def prepare_sun_classification(
    src: Path | str,
    dest: Path | str,
    seed: int = 0,
    stratify_morphology: bool = False,
) -> None:
    """Prepare SUN dataset for binary classification.

    Splits are generated on the case level and balanced so that each split
    contains the same number of positive and negative cases.

    Parameters
    ----------
    src:
        Path to canonical SUN dataset (with images/ and manifests/).
    dest:
        Output directory for classification dataset.
    seed:
        Random seed for deterministic sampling.
    stratify_morphology:
        If ``True``, positive cases are split stratified by Paris morphology
        groups (flat vs polypoid).
    """

    src = Path(src)
    dest = Path(dest)
    frames_df = _load_metadata(src)

    pos_cases = _build_positive_cases(frames_df, seed)
    neg_cases = _build_negative_cases(frames_df)

    # Assign synthetic sequential IDs
    all_cases: List[_Case] = []
    counter = 1
    for case in pos_cases + neg_cases:
        case.new_id = f"case{counter:03d}"
        counter += 1
        all_cases.append(case)

    pos_cases_new = [c for c in all_cases if c.has_polyp]
    neg_cases_new = [c for c in all_cases if not c.has_polyp]

    random.Random(seed).shuffle(pos_cases_new)
    random.Random(seed + 1).shuffle(neg_cases_new)

    # Balance splits by truncating the majority class so that positives and
    # negatives contribute the same number of cases.
    n_balanced = min(len(pos_cases_new), len(neg_cases_new))
    pos_cases_new = pos_cases_new[:n_balanced]
    neg_cases_new = neg_cases_new[:n_balanced]

    if stratify_morphology:
        groups: Dict[str | None, List[_Case]] = {}
        for case in pos_cases_new:
            groups.setdefault(case.morphology, []).append(case)
        pos_train: List[_Case] = []
        pos_val: List[_Case] = []
        pos_test: List[_Case] = []
        for g_cases in groups.values():
            random.Random(seed).shuffle(g_cases)
            tr, va, te = _split_cases(g_cases)
            pos_train.extend(tr)
            pos_val.extend(va)
            pos_test.extend(te)
        random.Random(seed).shuffle(pos_train)
        random.Random(seed).shuffle(pos_val)
        random.Random(seed).shuffle(pos_test)
        n_train = len(pos_train)
        n_val = len(pos_val)
        n_test = len(pos_test)
        neg_train = neg_cases_new[:n_train]
        neg_val = neg_cases_new[n_train : n_train + n_val]
        neg_test = neg_cases_new[n_train + n_val : n_train + n_val + n_test]
    else:
        pos_train, pos_val, pos_test = _split_cases(pos_cases_new)
        neg_train, neg_val, neg_test = _split_cases(neg_cases_new)

    splits = {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test,
    }

    for split_name, cases in splits.items():
        split_dest = dest / split_name
        for case in cases:
            _copy_case(case, src, split_dest)

def create_sun_classification_subset(
    src: Path | str,
    dest: Path | str,
    fraction: float,
    seed: int = 0,
) -> None:
    """Create a balanced training subset from a prepared dataset.

    The ``src`` directory is expected to follow the layout produced by
    :func:`prepare_sun_classification`.  Only the training split is
    subsampled at the *case* level; validation and test splits are copied
    unchanged.  Positive and negative cases are sampled in equal numbers.

    Parameters
    ----------
    src:
        Path to the source dataset prepared with
        :func:`prepare_sun_classification`.
    dest:
        Output directory for the subset dataset.
    fraction:
        Fraction of training cases to include (``0 < fraction <= 1``).
    seed:
        Random seed controlling which cases are selected.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the (0, 1] range")

    src = Path(src)
    dest = Path(dest)
    rng = random.Random(seed)

    classes = ["polyp", "non-polyp"]
    selected: dict[str, list[str]] = {}

    for cls in classes:
        img_paths = list((src / "train" / cls).glob("*"))
        case_ids = sorted({p.stem.split("_")[0] for p in img_paths})
        total = len(case_ids)
        n = max(1, round(fraction * total))
        n = min(n, total)
        selected[cls] = rng.sample(case_ids, n)

    for split in ["val", "test"]:
        for cls in classes:
            src_dir = src / split / cls
            dst_dir = dest / split / cls
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    for cls in classes:
        src_dir = src / "train" / cls
        dst_dir = dest / "train" / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for case_id in selected[cls]:
            for img in src_dir.glob(f"{case_id}_*"):
                shutil.copy(img, dst_dir / img.name)


__all__ = ["prepare_sun_classification", "create_sun_classification_subset"]
