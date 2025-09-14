import yaml
from polypdb.manifests.core import ManifestRow, sha256_file
from polypdb.manifests.io import read_csv, write_pack, validate_pack


def test_roundtrip_and_hash(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (root / "images" / name).write_bytes(b"")

    rows = [
        ManifestRow("ds", "train", "c1", "f1", "images/a.jpg", 1, "local", "polypoid"),
        ManifestRow("ds", "train", "c2", "f2", "images/b.jpg", 0, "local", "flat"),
        ManifestRow("ds", "val", "c3", "f3", "images/c.jpg", 1, "local", ""),
    ]

    splits = {"train": rows[:2], "val": rows[2:]}
    meta = {
        "name": "test-pack",
        "version": "0.0",
        "generator": {
            "repo": "repo",
            "git_commit": "deadbeef",
            "command": "cmd",
            "seed": 0,
        },
        "policy": {
            "unit_of_split": "case",
            "frames_per_case": 1,
            "balance": "none",
            "selection": "all",
        },
        "roots": {"local": str(root)},
    }

    selection_key_policy = {
        "granularity": "case",
        "key_field": "frame_id",
        "construction": "dataset+case_id+frame_id",
        "normalization": ["lower"],
    }

    out_dir = tmp_path / "pack"
    write_pack(out_dir, splits, meta, selection_key_policy)

    roundtrip = read_csv(out_dir / "train.csv")
    assert roundtrip == rows[:2]

    manifest = yaml.safe_load((out_dir / "manifest.yaml").read_text())
    assert manifest["hashes"]["train"] == sha256_file(out_dir / "train.csv")
    assert manifest["selection_key_policy"]["granularity"] == "case"

    manifest_rt = validate_pack(out_dir, require_files_exist=True)
    assert manifest_rt["selection_key_policy"] == selection_key_policy


def test_validate_pack_unique_key(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    for name in ["a.jpg", "b.jpg"]:
        (root / "images" / name).write_bytes(b"")

    rows = [
        ManifestRow("ds", "train", "c1", "f", "images/a.jpg", 1, "local", "polypoid", variant="v1"),
        ManifestRow("ds", "train", "c1", "f", "images/b.jpg", 0, "local", "flat", variant="v2"),
    ]

    splits = {"train": rows}
    meta = {
        "name": "test-pack",
        "version": "0.0",
        "roots": {"local": str(root)},
        "assertions": {"unique_key": ["case_id", "frame_id", "variant"]},
    }

    out_dir = tmp_path / "pack"
    write_pack(out_dir, splits, meta)

    # Should not raise despite duplicate frame_id because unique key includes variant
    validate_pack(out_dir, require_files_exist=True)
