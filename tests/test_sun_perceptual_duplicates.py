import pandas as pd
from PIL import Image

from polypdb.preprocess.sun import _drop_duplicate_frames


def test_near_duplicate_prefers_lossless(tmp_path):
    img = Image.new("RGB", (10, 10), color="red")
    lossless = tmp_path / "a.png"
    lossy = tmp_path / "b.jpg"
    img.save(lossless)
    img.save(lossy, quality=50)

    df = pd.DataFrame(
        {
            "SRC_PATH": [str(lossless), str(lossy)],
            "WIDTH": [10, 10],
            "HEIGHT": [10, 10],
        }
    )

    result = _drop_duplicate_frames(df)
    assert len(result) == 1
    assert result["SRC_PATH"].iloc[0] == str(lossless)

