from pathlib import Path

import pytest
import yaml

from polypdb.fewshot.polypgen_fewshot import _load_frame_metadata
from polypdb.manifests.core import ManifestRow
from polypdb.manifests.io import validate_pack, write_pack
from polypdb.manifests.provenance import tool_provenance


def _row(split: str, case: str, frame: str) -> ManifestRow:
    return ManifestRow(
        dataset="SUN", split=split, case_id=case, frame_id=frame,
        frame_path=f"images/{frame}.jpg", label=1, store_id="sun",
    )


def test_manifest_records_provenance(tmp_path: Path) -> None:
    root = tmp_path / "root" / "manifests"
    root.mkdir(parents=True)
    (root / "frames.parquet").write_bytes(b"frames")
    out = tmp_path / "pack"
    write_pack(out, {"test": [_row("test", "A", "a1")]}, {"name": "t", "roots": {"sun": str(tmp_path / "root")}})
    m = yaml.safe_load((out / "manifest.yaml").read_text())
    prov = tool_provenance()
    assert m["generator"]["git_commit"] == prov["git_commit"]
    assert "git_dirty" in m["generator"]
    assert set(m["generator"]["environment"]) == {"python", "pandas", "numpy"}
    assert m["inputs"]["sun"]["frames_parquet_sha256"]
    assert m["inputs"]["sun"]["cases_parquet_sha256"] is None


def test_fewshot_metadata_fails_without_frames_parquet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_frame_metadata({"polypgen_clean": str(tmp_path / "missing")}, {"polypgen_clean"}, {"f1"})
    with pytest.raises(FileNotFoundError):
        _load_frame_metadata({}, {"polypgen_clean"}, {"f1"})


def test_validate_pack_detects_shared_negative_video(tmp_path: Path) -> None:
    def neg(split, case, frame):
        return ManifestRow(
            dataset="SUN", split=split, case_id=case, frame_id=frame, frame_path=f"images/{frame}.jpg",
            label=0, store_id="sun", synthetic_case_id=case, source_case_id="NEG_SUN_case1",
            synthetic_case=True,
        )
    splits = {"train": [neg("train", "NEG_v0", "f0")], "val": [neg("val", "NEG_v1", "f1")]}
    with pytest.raises(AssertionError, match="negative source video"):
        write_pack(tmp_path / "p", splits, {"name": "t"})
