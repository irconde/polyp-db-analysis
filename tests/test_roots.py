from pathlib import Path
import json
import yaml

from polypdb.manifests.core import ManifestRow
from polypdb.utils.roots import load_roots, resolve_path


def test_load_roots_priority(tmp_path: Path) -> None:
    manifest_yaml = tmp_path / "manifest.yaml"
    manifest_yaml.write_text(yaml.safe_dump({"roots": {"a": "/m/a", "b": "/m/b"}}))
    roots_json = tmp_path / "roots.json"
    roots_json.write_text(json.dumps({"b": "/c/b", "c": "/c/c"}))

    roots = load_roots(roots_json, manifest_yaml, Path("/d/default"))
    assert roots["a"] == "/m/a"
    assert roots["b"] == "/c/b"
    assert roots["c"] == "/c/c"
    assert roots["*"] == "/d/default"


def test_resolve_path(tmp_path: Path) -> None:
    roots = {"*": str(tmp_path)}
    row = ManifestRow(
        dataset="ds",
        split="train",
        case_id="c",
        frame_id="f",
        frame_path="image.jpg",
        label=1,
        store_id="unknown",
    )

    expected = tmp_path / "image.jpg"
    assert resolve_path(row, roots) == expected

    # Missing root should raise
    try:
        resolve_path(row, {})
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
