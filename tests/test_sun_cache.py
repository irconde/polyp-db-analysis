import pandas as pd
from PIL import Image, PngImagePlugin
from polypdb.preprocess.sun import _drop_duplicate_frames


def test_drop_duplicate_frames_uses_cache(monkeypatch, tmp_path):
    img = Image.new("RGB", (10, 10), color="red")
    path1 = tmp_path / "a.png"
    path2 = tmp_path / "b.png"
    img.save(path1)
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", "x" * 100)
    img.save(path2, pnginfo=info)

    df = pd.DataFrame(
        {
            "SRC_PATH": [str(path1), str(path2)],
            "WIDTH": [10, 10],
            "HEIGHT": [10, 10],
        }
    )

    cache_file = tmp_path / "cache.csv"
    result = _drop_duplicate_frames(df, cache_file=str(cache_file))
    assert len(result) == 1
    assert cache_file.exists()

    calls = 0

    def fake_hash(path):
        nonlocal calls
        calls += 1
        return "unused"

    monkeypatch.setattr("polypdb.preprocess.sun._pixel_hash", fake_hash)

    result2 = _drop_duplicate_frames(df, cache_file=str(cache_file))
    assert len(result2) == 1
    assert calls == 0
