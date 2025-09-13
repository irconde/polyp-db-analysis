from pathlib import Path

from PIL import Image
import yaml
import pandas as pd

from polypdb.datasets.polypgen_build import build_polypgen_eval
from polypdb.manifests.io import read_csv


def test_build_polypgen_eval(tmp_path: Path) -> None:
    root = tmp_path / "pg"
    img_dir = root / "images"
    man_dir = root / "manifests"
    img_dir.mkdir(parents=True)
    man_dir.mkdir(parents=True)
    for name, size in zip(["pos.png", "neg.png"], [(2, 3), (4, 5)]):
        Image.new("RGB", size).save(img_dir / name)

    ann = pd.DataFrame(
        {
            "img_path": ["images/pos.png", "images/neg.png"],
            "has_polyp": [1, 0],
            "global_frame_id": ["gf_pos", "gf_neg"],
            "center_id": ["c1", "c2"],
            "img_width": [2, 4],
            "img_height": [3, 5],
        }
    )
    ann.to_parquet(man_dir / "frames.parquet", index=False)

    out = tmp_path / "out"
    build_polypgen_eval(root, out, track="clean")

    rows = read_csv(out / "eval.csv")
    assert len(rows) == 2
    assert all(r.dataset == "PolypGen" for r in rows)
    assert all(r.split == "eval" for r in rows)
    assert all(r.store_id == "polypgen_clean" for r in rows)
    labels = {r.frame_id: r.label for r in rows}
    cases = {r.frame_id: r.case_id for r in rows}
    assert labels["gf_pos"] == 1
    assert labels["gf_neg"] == 0
    assert cases["gf_pos"] == "c1"
    assert cases["gf_neg"] == "c2"
    sizes = {r.frame_id: (r.width, r.height) for r in rows}
    assert sizes["gf_pos"] == (2, 3)
    assert sizes["gf_neg"] == (4, 5)

    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["policy"]["unit_of_split"] == "frame"
    assert manifest["roots"]["polypgen_clean"] == str(root)
    assert manifest["assertions"]["label_counts"] == {"pos": 1, "neg": 1}
    summary = {s["split"]: s for s in manifest["summary_by_split"]}
    assert summary["eval"]["label_counts"] == {"pos": 1, "neg": 1}
