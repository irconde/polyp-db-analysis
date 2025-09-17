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
    specs = [
        ("pos.png", (2, 3)),
        ("neg.png", (4, 5)),
        ("neg_seq.png", (6, 7)),
    ]
    for name, size in specs:
        Image.new("RGB", size).save(img_dir / name)

    ann = pd.DataFrame(
        {
            "img_path": [f"images/{name}" for name, _ in specs],
            "has_polyp": [1, 0, 0],
            "global_frame_id": ["gf_pos", "gf_neg", "gf_neg_seq"],
            "center_id": ["c1", "c2", ""],
            "img_width": [w for _, (w, _) in specs],
            "img_height": [h for _, (_, h) in specs],
            "origin": ["frame", "frame", "neg_seq"],
            "sequence_id": [pd.NA, pd.NA, "seqA"],
        }
    )
    ann.to_parquet(man_dir / "frames.parquet", index=False)

    out = tmp_path / "out"
    build_polypgen_eval(root, out, track="clean")

    rows = read_csv(out / "test.csv")
    assert len(rows) == 2
    assert {r.frame_id for r in rows} == {"gf_pos", "gf_neg"}
    assert all(r.dataset == "PolypGen" for r in rows)
    assert all(r.split == "test" for r in rows)
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
    assert summary["test"]["label_counts"] == {"pos": 1, "neg": 1}


def test_build_polypgen_eval_with_sequence_negatives(tmp_path: Path) -> None:
    root = tmp_path / "pg_ext"
    img_dir = root / "images"
    man_dir = root / "manifests"
    img_dir.mkdir(parents=True)
    man_dir.mkdir(parents=True)

    specs = [
        ("pos.png", (2, 3)),
        ("neg.png", (4, 5)),
        ("neg_seq.png", (6, 7)),
    ]
    for name, size in specs:
        Image.new("RGB", size).save(img_dir / name)

    ann = pd.DataFrame(
        {
            "img_path": [f"images/{name}" for name, _ in specs],
            "has_polyp": [1, 0, 0],
            "global_frame_id": ["gf_pos", "gf_neg", "gf_neg_seq"],
            "center_id": ["c1", "c2", ""],
            "img_width": [w for _, (w, _) in specs],
            "img_height": [h for _, (_, h) in specs],
            "origin": ["frame", "frame", "neg_seq"],
            "sequence_id": [pd.NA, pd.NA, "seqA"],
        }
    )
    ann.to_parquet(man_dir / "frames.parquet", index=False)

    out = tmp_path / "out_ext"
    build_polypgen_eval(
        root,
        out,
        track="clean",
        include_negative_sequences=True,
    )

    rows = read_csv(out / "test.csv")
    assert len(rows) == 3
    assert {r.frame_id for r in rows} == {"gf_pos", "gf_neg", "gf_neg_seq"}

    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["name"] == "polypgen-clean-test-extended"
    assert manifest["assertions"]["label_counts"] == {"pos": 1, "neg": 2}
