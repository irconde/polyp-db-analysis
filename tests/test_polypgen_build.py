from pathlib import Path

from PIL import Image
import yaml

from polypdb.datasets.polypgen_build import build_polypgen_eval
from polypdb.manifests.io import read_csv


def test_build_polypgen_eval(tmp_path: Path) -> None:
    root = tmp_path / "pg"
    root.mkdir()
    for name in ["a.png", "b.png"]:
        Image.new("RGB", (1, 1)).save(root / name)

    out = tmp_path / "out"
    build_polypgen_eval(root, out, track="clean")

    rows = read_csv(out / "eval.csv")
    assert len(rows) == 2
    assert all(r.dataset == "PolypGen" for r in rows)
    assert all(r.split == "eval" for r in rows)
    assert all(r.store_id == "polypgen_clean" for r in rows)
    assert {r.frame_id for r in rows} == {"a", "b"}

    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["policy"]["unit_of_split"] == "frame"
    assert manifest["roots"]["polypgen_clean"] == str(root)
