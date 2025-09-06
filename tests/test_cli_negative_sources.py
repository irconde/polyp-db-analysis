import sys
import types
import yaml
import importlib


def test_pipeline_passes_negative_sources(tmp_path, monkeypatch):
    sys.modules['polypdb.preprocess.polypgen'] = types.SimpleNamespace(
        convert_to_standard_layout=lambda *a, **k: None
    )
    cli = importlib.import_module('polypdb.cli')

    cfg = {
        'raw_sources': {},
        'base_path': str(tmp_path),
        'datasets': [],
        'num_elements': None,
        'prepare': False,
        'negative_sources': {'sun': 'sun_negatives'},
    }
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text(yaml.dump(cfg))

    captured = {}

    def fake_merge_datasets(cfg):
        captured['cfg'] = cfg

    monkeypatch.setattr(cli, 'merge_datasets', fake_merge_datasets)
    monkeypatch.setattr(sys, 'argv', ['prog', 'pipeline', str(cfg_path)])
    cli.main()

    assert captured['cfg'].negative_sources == {'sun': 'sun_negatives'}
