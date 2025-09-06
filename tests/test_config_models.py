import pytest

from polypdb.cli.common import load_config
from polypdb.config_models import MergeConfig, PrepareConfig


def test_merge_config_requires_base_path(tmp_path):
    cfg_path = tmp_path / "merge.yaml"
    cfg_path.write_text("datasets: ['a']\n")
    with pytest.raises(ValueError, match="base_path"):
        load_config(str(cfg_path), MergeConfig)


def test_prepare_config_requires_splitting(tmp_path):
    cfg_text = (
        "src_path: /data\n"
        "classification_type: BINARY\n"
        "negative_sampling:\n  ratio: 0.5\n  distribution: SPLIT-RATIO\n"
    )
    cfg_path = tmp_path / "prep.yaml"
    cfg_path.write_text(cfg_text)
    with pytest.raises(ValueError, match="splitting_options"):
        load_config(str(cfg_path), PrepareConfig)
