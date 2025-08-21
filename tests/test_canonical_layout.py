import pandas as pd
from PIL import Image

from polypdb.merge import copy_and_rename_images


def test_copy_and_rename_images_canonical_naming(tmp_path):
    base_path = tmp_path

    # create raw image
    raw_dir = base_path / "raw" / "sun" / "img"
    raw_dir.mkdir(parents=True)
    img_path = raw_dir / "frame1.JPG"
    Image.new("RGB", (10, 10)).save(img_path)

    df = pd.DataFrame({
        "ORIGINAL DATASET": ["sun"],
        "POLYP ID": ["SUN_0001"],
        "FRAME": ["frame1.JPG"],
    })

    canonical_dir = base_path / "canonical" / "images"
    copy_and_rename_images(df, str(base_path), str(canonical_dir))

    expected_name = "sun.SUN_0001.000001.jpg"
    assert df.loc[0, "FRAME"] == expected_name
    assert df.loc[0, "global_frame_id"] == "sun.SUN_0001.000001"
    assert df.loc[0, "raw_path"] == str(img_path)
    assert (canonical_dir / expected_name).exists()
