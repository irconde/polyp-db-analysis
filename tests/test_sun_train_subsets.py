from pathlib import Path
from collections import Counter
import yaml

from polypdb.datasets.sun_build import build_sun_full
from polypdb.subsets.sun_subsets import (
    CANONICAL_SUBSET_PERCENTAGES,
    CANONICAL_SUBSET_SEED,
    build_sun_canonical_subsets,
    build_sun_train_subsets,
)
from polypdb.manifests.io import read_csv


def _make_index():
    index = {}
    for i in range(20):
        cid = f"p{i:02d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(40)
        ]
        index[cid] = {"label": 1, "frames": frames}
    for i in range(20):
        cid = f"n{i:02d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(40)
        ]
        index[cid] = {"label": 0, "frames": frames}
    return index


def test_build_sun_train_subsets(tmp_path: Path) -> None:
    index = _make_index()

    def indexer(_root: Path):
        return index

    full_pack = tmp_path / "sun_full"
    build_sun_full(Path("/data"), full_pack, seed=0, indexer=indexer, verify=False)

    out_dir = tmp_path
    build_sun_train_subsets(full_pack, percents=[50], seeds=[0, 1], out_dir=out_dir)

    full_rows = read_csv(full_pack / "train.csv")
    full_ids = {(r.case_id, r.frame_id) for r in full_rows}

    for seed in [0, 1]:
        pack_dir = out_dir / f"sun_p50_seed{seed}"
        subset_rows = read_csv(pack_dir / "train.csv")
        subset_ids = {(r.case_id, r.frame_id) for r in subset_rows}
        assert subset_ids < full_ids
        counts = Counter(r.case_id for r in subset_rows)
        assert counts and all(v == 33 for v in counts.values())
        pos_case_ids = {r.case_id for r in subset_rows if r.label == 1}
        neg_case_ids = {r.case_id for r in subset_rows if r.label == 0}
        assert not (pack_dir / "val.csv").exists()
        assert not (pack_dir / "test.csv").exists()
        manifest_full = yaml.safe_load((full_pack / "manifest.yaml").read_text(encoding="utf-8"))
        manifest_subset = yaml.safe_load((pack_dir / "manifest.yaml").read_text(encoding="utf-8"))
        counts_train = manifest_subset["counts"]["train"]
        assert counts_train["pos_cases"] == len(pos_case_ids)
        assert counts_train["neg_cases"] == len(neg_case_ids)
        assert counts_train["pos"] == sum(1 for r in subset_rows if r.label == 1)
        assert counts_train["neg"] == sum(1 for r in subset_rows if r.label == 0)
        assert manifest_subset["counts"]["val"] == manifest_full["counts"]["val"]
        assert manifest_subset["hashes"]["test"] == manifest_full["hashes"]["test"]
        assert manifest_subset["parent_pack"]["name"] == "sun_full"
        assert manifest_subset["parent_pack"]["hashes"]["val"] == manifest_full["hashes"]["val"]
        assert manifest_subset["parent_pack"]["hashes"]["test"] == manifest_full["hashes"]["test"]
        assert manifest_subset["parent_pack"]["hashes"]["train"] == manifest_full["hashes"]["train"]
        assert manifest_subset["parent_pack"]["path"] == str(full_pack)
        assert manifest_subset["subset"] == {
            "percent": 50,
            "seed": seed,
            "source_pool": "train",
        }
        assert manifest_subset["assertions"]["subset_train_from_parent_train"] is True
        assert manifest_subset["assertions"]["subset_train_pos_eq_neg"] is (
            len(pos_case_ids) == len(neg_case_ids)
        )
        assert manifest_subset["assertions"]["val_test_hash_match_parent"] is True
        assert manifest_subset["policy"]["selection_scope"] == "case"
        assert manifest_subset["frame_sampling"]["selection_scope"] == "frame"


def test_build_sun_canonical_subsets(tmp_path: Path) -> None:
    index = _make_index()

    def indexer(_root: Path):
        return index

    full_pack = tmp_path / "sun_full"
    build_sun_full(Path("/data"), full_pack, seed=0, indexer=indexer, verify=False)

    out_dir = tmp_path / "canon"
    build_sun_canonical_subsets(full_pack, out_dir)

    for percent in CANONICAL_SUBSET_PERCENTAGES:
        pack_dir = out_dir / f"sun_p{percent:02d}"
        assert pack_dir.exists()
        manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text(encoding="utf-8"))
        assert manifest["name"] == f"sun_p{percent:02d}"
        subset_meta = manifest["subset"]
        assert subset_meta["percent"] == percent
        assert subset_meta["seed"] == CANONICAL_SUBSET_SEED
        assert subset_meta["source_pool"] == "train"
        assert subset_meta.get("canonical") is True
        generator = manifest["generator"]
        assert generator.get("percent_seed_map", {}).get(percent) == CANONICAL_SUBSET_SEED
        assert "subsets-canonical" in generator["command"]

    # canonical call should still obey reproducibility when invoked again
    build_sun_canonical_subsets(full_pack, out_dir)
    for percent in CANONICAL_SUBSET_PERCENTAGES:
        pack_dir = out_dir / f"sun_p{percent:02d}"
        assert pack_dir.exists()


def test_build_sun_train_subsets_single_seed(tmp_path: Path) -> None:
    index = _make_index()

    def indexer(_root: Path):
        return index

    full_pack = tmp_path / "sun_full"
    build_sun_full(Path("/data"), full_pack, seed=0, indexer=indexer, verify=False)

    out_dir = tmp_path / "single"
    build_sun_train_subsets(full_pack, percents=[25, 50], seeds=[7], out_dir=out_dir)

    for percent in (25, 50):
        pack_dir = out_dir / f"sun_p{percent:02d}"
        assert pack_dir.exists()
        manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text(encoding="utf-8"))
        assert manifest["name"] == f"sun_p{percent:02d}"
        subset_meta = manifest["subset"]
        assert subset_meta["seed"] == 7
        assert subset_meta["percent"] == percent
