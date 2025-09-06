import sys
import types
import yaml
import pytest
import importlib


def test_pipeline_single_dataset_filters_sources(tmp_path, monkeypatch):
    sys.modules['polypdb.preprocess.polypgen'] = types.SimpleNamespace(
        convert_to_standard_layout=lambda *a, **k: None
    )
    cli = importlib.import_module('polypdb.cli')
    pipeline_mod = importlib.import_module('polypdb.cli.pipeline')

    cfg = {
        'raw_sources': {'polypgen': '/unused/path'},
        'base_path': str(tmp_path),
        'datasets': ['polypgen', 'polar'],
        'num_elements': None,
        'prepare': False,
        'negative_sources': {'polypgen': 'pg_neg', 'polar': 'po_neg'},
    }
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text(yaml.dump(cfg))

    captured = {}

    def fake_merge_datasets(cfg):
        captured['cfg'] = cfg

    fake_converters = {
        'polypgen': lambda *args, **kwargs: types.SimpleNamespace(convert=lambda: None)
    }

    monkeypatch.setattr(cli, 'merge_datasets', fake_merge_datasets)
    monkeypatch.setattr(pipeline_mod, 'CONVERTERS', fake_converters)
    monkeypatch.setattr(
        sys,
        'argv',
        ['prog', 'pipeline', '--dataset', 'polypgen', str(cfg_path)],
    )
    cli.main()

    assert captured['cfg'].datasets == ['polypgen']
    assert captured['cfg'].negative_sources == {'polypgen': 'pg_neg'}


def test_pipeline_unknown_dataset_errors(tmp_path, monkeypatch):
    sys.modules['polypdb.preprocess.polypgen'] = types.SimpleNamespace(
        convert_to_standard_layout=lambda *a, **k: None
    )
    cli = importlib.import_module('polypdb.cli')
    pipeline_mod = importlib.import_module('polypdb.cli.pipeline')

    cfg = {
        'raw_sources': {'polypgen': '/unused/path'},
        'base_path': str(tmp_path),
        'datasets': ['polypgen'],
        'num_elements': None,
        'prepare': False,
    }
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text(yaml.dump(cfg))

    fake_converters = {
        'polypgen': lambda *args, **kwargs: types.SimpleNamespace(convert=lambda: None),
        'polar': lambda *args, **kwargs: types.SimpleNamespace(convert=lambda: None),
    }
    monkeypatch.setattr(pipeline_mod, 'CONVERTERS', fake_converters)
    monkeypatch.setattr(
        sys,
        'argv',
        ['prog', 'pipeline', '--dataset', 'polar', str(cfg_path)],
    )
    with pytest.raises(ValueError, match="Dataset 'polar' not defined"):
        cli.main()
