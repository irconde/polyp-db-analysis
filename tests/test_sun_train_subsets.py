from pathlib import Path
from collections import Counter
import yaml

from polypdb.datasets.sun_build import build_sun_full
from polypdb.subsets.sun_subsets import build_sun_train_subsets
from polypdb.manifests.io import read_csv


def _make_index():
    index = {}
    for i in range(2):
        cid = f"p{i:02d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(40)
        ]
        index[cid] = {"label": 1, "frames": frames}
    for i in range(2):
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
        assert not (pack_dir / "val.csv").exists()
        assert not (pack_dir / "test.csv").exists()
        manifest_full = yaml.safe_load((full_pack / "manifest.yaml").read_text())
        manifest_subset = yaml.safe_load((pack_dir / "manifest.yaml").read_text())
        assert manifest_subset["counts"]["val"] == manifest_full["counts"]["val"]
        assert manifest_subset["hashes"]["test"] == manifest_full["hashes"]["test"]
