from pathlib import Path
import yaml

from polypdb.utils.selection_key_policy import load_selection_key_policy


def test_load_selection_key_policy(tmp_path: Path) -> None:
    man = tmp_path / "manifest.yaml"
    policy = {"granularity": "case", "key_field": "frame_id"}
    man.write_text(yaml.safe_dump({"selection_key_policy": policy}))
    assert load_selection_key_policy(man) == policy


def test_load_selection_key_policy_missing(tmp_path: Path) -> None:
    man = tmp_path / "manifest.yaml"
    man.write_text(yaml.safe_dump({}))
    assert load_selection_key_policy(man) is None
    assert load_selection_key_policy(None) is None
