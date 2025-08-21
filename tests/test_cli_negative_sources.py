import sys
import yaml

from polypdb import cli


def test_pipeline_passes_negative_sources(tmp_path, monkeypatch):
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

    def fake_merge_datasets(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, 'merge_datasets', fake_merge_datasets)
    monkeypatch.setattr(sys, 'argv', ['prog', 'pipeline', str(cfg_path)])
    cli.main()

    assert captured['negative_sources'] == {'sun': 'sun_negatives'}
