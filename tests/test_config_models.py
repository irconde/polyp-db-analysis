import pytest

from polypdb.cli.common import load_config
from polypdb.config_models import MergeConfig, PrepareConfig


def test_merge_config_requires_base_path(tmp_path):
    cfg_path = tmp_path / "merge.yaml"
    cfg_path.write_text("datasets: ['a']\n")
    with pytest.raises(ValueError, match="base_path"):
        load_config(str(cfg_path), MergeConfig)


def test_merge_config_default_write_cases_true(tmp_path):
    cfg_text = "base_path: /data\ndatasets: ['a']\n"
    cfg_path = tmp_path / "merge.yaml"
    cfg_path.write_text(cfg_text)
    cfg = load_config(str(cfg_path), MergeConfig)
    assert cfg.write_cases is True


def test_merge_config_respects_write_cases_flag(tmp_path):
    cfg_text = "base_path: /data\ndatasets: ['a']\nwrite_cases: false\n"
    cfg_path = tmp_path / "merge.yaml"
    cfg_path.write_text(cfg_text)
    cfg = load_config(str(cfg_path), MergeConfig)
    assert cfg.write_cases is False


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


def test_splitting_options_require_ratios_or_counts(tmp_path):
    cfg_text = (
        "src_path: /data\n"
        "classification_type: BINARY\n"
        "splitting_options:\n"
        "  split_type: RANDOM\n"
        "  split_level: FRAME\n"
        "negative_sampling:\n  ratio: 0.5\n  distribution: SPLIT-RATIO\n"
    )
    cfg_path = tmp_path / "prep.yaml"
    cfg_path.write_text(cfg_text)
    with pytest.raises(ValueError, match="either ratios or counts"):
        load_config(str(cfg_path), PrepareConfig)


def test_prepare_config_accepts_counts(tmp_path):
    cfg_text = (
        "src_path: /data\n"
        "classification_type: BINARY\n"
        "splitting_options:\n"
        "  train_count: 1\n"
        "  validation_count: 0\n"
        "  split_type: RANDOM\n"
        "  split_level: FRAME\n"
        "negative_sampling:\n  ratio: 0.5\n  distribution: SPLIT-RATIO\n"
    )
    cfg_path = tmp_path / "prep.yaml"
    cfg_path.write_text(cfg_text)
    cfg = load_config(str(cfg_path), PrepareConfig)
    assert cfg.splitting_options.train_count == 1
