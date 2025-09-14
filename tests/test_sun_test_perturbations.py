from pathlib import Path
from collections import defaultdict
import yaml

from polypdb.datasets.sun_build import build_sun_full
from polypdb.datasets.sun_perturbations import (
    PERTURBATIONS,
    build_sun_test_perturbations,
    _derive_seed,
)
from polypdb.manifests.io import read_csv


def _make_index(n_pos: int = 100, n_neg: int = 100, frames_per_case: int = 33):
    index = {}
    for i in range(n_pos):
        cid = f"p{i:03d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(frames_per_case)
        ]
        index[cid] = {"label": 1, "frames": frames}
    for i in range(n_neg):
        cid = f"n{i:03d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(frames_per_case)
        ]
        index[cid] = {"label": 0, "frames": frames}
    return index


def test_build_sun_test_perturbations(tmp_path):
    index = _make_index()

    def indexer(_root: Path):
        return index

    full_pack = tmp_path / "sun_full"
    build_sun_full(Path("/data"), full_pack, seed=0, indexer=indexer, verify=False)

    pert_pack = tmp_path / "sun_test_perturbations"
    build_sun_test_perturbations(full_pack, pert_pack)

    rows = read_csv(pert_pack / "test.csv")

    assert len(rows) == 15840
    orig_ids = [r.orig_frame_id for r in rows]
    assert len(set(orig_ids)) == 990

    variants = defaultdict(set)
    labels = defaultdict(set)
    cases = defaultdict(set)
    for r in rows:
        variants[r.orig_frame_id].add(r.perturbation_id)
        labels[r.orig_frame_id].add(r.label)
        cases[r.orig_frame_id].add(r.case_id)
    assert all(len(v) == 16 for v in variants.values())
    assert all(len(v) == 1 for v in labels.values())
    assert all(len(v) == 1 for v in cases.values())

    for r in rows[:5]:
        assert r.rng_seed == _derive_seed(47, r.orig_frame_id, r.perturbation_id)

    manifest = yaml.safe_load((pert_pack / "manifest.yaml").read_text())
    assertions = manifest["assertions"]
    assert manifest["perturbations"] == PERTURBATIONS
    assert assertions["per_condition_row_count"] == 990
    assert assertions["total_rows"] == 15840
    assert assertions["conditions_len"] == 16
    assert assertions["label_invariant_across_conditions"]
    assert assertions["no_case_leakage_across_conditions"]
    assert assertions["clean_condition_present"]
