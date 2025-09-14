"""Tests for virtual negative case generation in :func:`build_sun_full`."""
from collections import Counter
from pathlib import Path

from polypdb.datasets.sun_build import build_sun_full
from polypdb.manifests.io import read_csv
import yaml


def _make_index():
    index = {}
    # 100 positive cases each with exactly 33 frames
    for i in range(100):
        cid = f"p{i:03d}"
        frames = [
            {"frame_path": f"{cid}/f{j:03d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(33)
        ]
        index[cid] = {"label": 1, "frames": frames}

    # 13 long negative cases each with 330 frames (10 chunks of 33)
    for i in range(13):
        cid = f"n{i:02d}"
        frames = [
            {"frame_path": f"{cid}/f{j:04d}.jpg", "frame_index": j, "width": 1, "height": 1}
            for j in range(330)
        ]
        index[cid] = {"label": 0, "frames": frames}

    return index


def test_virtual_negative_cases(tmp_path):
    index = _make_index()

    def indexer(_root: Path):
        return index

    build_sun_full(Path("/data"), tmp_path, seed=0, indexer=indexer, verify=False)

    splits = {split: read_csv(tmp_path / f"{split}.csv") for split in ["train", "val", "test"]}

    for split, expected in [("train", 70), ("val", 15), ("test", 15)]:
        rows = splits[split]
        pos_cases = {r.case_id for r in rows if r.label == 1}
        neg_cases = {r.case_id for r in rows if r.label == 0}
        assert len(pos_cases) == expected
        counts = Counter(r.case_id for r in rows)
        assert all(v == 33 for v in counts.values())
        for r in rows:
            if r.label == 0:
                assert r.split_unit == "case"
                assert r.synthetic_case is True

    all_neg_cases = {r.case_id for rows in splits.values() for r in rows if r.label == 0}
    assert len(all_neg_cases) == 100

    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assertions = manifest["assertions"]
    assert assertions["no_case_leakage_across_splits"] is True
    assert "no_negative_source_leakage" not in assertions
    expected_counts = {
        split: {
            "pos_cases": len({r.case_id for r in rows if r.label == 1}),
            "neg_cases": len({r.case_id for r in rows if r.label == 0}),
            "frames": len(rows),
        }
        for split, rows in splits.items()
    }
    assert assertions["split_targets_met"] == expected_counts
    assert assertions["all_cases_have_33_frames"] is True

    fs = manifest["frame_sampling"]
    assert fs == {
        "selection": "hmac_sha256(frame_path,salt)",
        "path_normalization": "posix,no-leading-slash,single-slash",
        "path_is_relative_to": "roots.sun",
        "salt_id": "seed-0",
        "tie_break": "frame_path",
        "frames_per_case": 33,
        "resampled_with_replacement": False,
    }

    neg_sources = manifest["neg_sources"]
    mapped_sources = {e["source_case_id"] for e in neg_sources}
    actual_sources = {
        r.source_case_id
        for rows in splits.values()
        for r in rows
        if r.label == 0
    }
    assert mapped_sources == actual_sources
    for entry in neg_sources:
        split = entry["split"]
        neg_cases = set(manifest["splits"][split]["neg_cases"])
        assert set(entry["synthetic_case_ids"]).issubset(neg_cases)
        assert len(entry["synthetic_case_ids"]) == len(entry["chunk_spans"])

    # check explicit split membership
    split_meta = manifest["splits"]
    for split, rows in splits.items():
        meta_pos = set(split_meta[split]["pos_cases"])
        meta_neg = set(split_meta[split]["neg_cases"])
        actual_pos = {r.case_id for r in rows if r.label == 1}
        actual_neg = {r.case_id for r in rows if r.label == 0}
        assert meta_pos == actual_pos
        assert meta_neg == actual_neg

    # summary table and case length histogram
    summary = {s["split"]: s for s in manifest["summary_by_split"]}
    for split, rows in splits.items():
        s = summary[split]
        assert s["frames"] == len(rows)
        assert s["pos_cases"] == len({r.case_id for r in rows if r.label == 1})
        assert s["neg_cases"] == len({r.case_id for r in rows if r.label == 0})
        assert s["pos_frames"] == sum(r.label == 1 for r in rows)
        assert s["neg_frames"] == sum(r.label == 0 for r in rows)

    assert manifest["case_lengths"] == {33: 200}

