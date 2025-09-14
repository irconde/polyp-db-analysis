from pathlib import Path
import yaml

from polypdb.fewshot import build_polypgen_fewshot
from polypdb.manifests.core import ManifestRow
from polypdb.manifests.io import read_csv, write_pack


def _make_test_pack(tmp_path: Path) -> Path:
    rows = []
    # create 10 positive and 10 negative rows
    for i in range(10):
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="test",
                case_id="",
                frame_id=f"pos{i}",
                frame_path=f"pos{i}.png",
                label=1,
                store_id="pg",
            )
        )
    for i in range(10):
        rows.append(
            ManifestRow(
                dataset="PolypGen",
                split="test",
                case_id="",
                frame_id=f"neg{i}",
                frame_path=f"neg{i}.png",
                label=0,
                store_id="pg",
            )
        )
    pack_dir = tmp_path / "test_pack"
    meta = {"name": "polypgen-clean-test", "version": "0.0", "policy": {"unit_of_split": "frame"}}
    write_pack(pack_dir, {"test": rows}, meta)
    return pack_dir


def test_polypgen_fewshot_disjoint_and_balanced(tmp_path: Path) -> None:
    test_pack = _make_test_pack(tmp_path)
    out_dir = tmp_path / "out"
    build_polypgen_fewshot(test_pack, [5], [123], out_dir)

    pack_dir = out_dir / "polypgen_fewshot_s5_seed123"
    train_rows = read_csv(pack_dir / "train.csv")
    test_rows = read_csv(pack_dir / "test.csv")

    # Disjointness
    train_ids = {r.frame_id for r in train_rows}
    test_ids = {r.frame_id for r in test_rows}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {r.frame_id for r in read_csv(test_pack / "test.csv")}

    # Approximate balance
    pos = sum(r.label == 1 for r in train_rows)
    neg = sum(r.label == 0 for r in train_rows)
    assert pos + neg == 5
    assert abs(pos - neg) <= 1

    manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text())
    assert manifest["policy"]["selection"] == "hmac_sha256(frame_id,123)"
    assert manifest["counts"]["train"]["frames"] == 5
    assert manifest["counts"]["test"]["frames"] == 20 - 5
