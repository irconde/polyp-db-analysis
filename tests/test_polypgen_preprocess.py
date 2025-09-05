import pandas as pd
from PIL import Image

from polypdb.preprocess.polypgen import convert_to_standard_layout


def test_convert_to_standard_layout_parses_bboxes(tmp_path, monkeypatch):
    # Create centre directories
    images_dir = tmp_path / "data_C1" / "images_C1"
    bbox_dir = tmp_path / "data_C1" / "bbox_C1"
    images_dir.mkdir(parents=True)
    bbox_dir.mkdir(parents=True)

    # Sample images
    Image.new("RGB", (10, 10)).save(images_dir / "frame1.jpg")
    Image.new("RGB", (8, 12)).save(images_dir / "frame2.jpg")

    # Bounding box annotations: frame1 has one box, frame2 none
    (bbox_dir / "frame1.txt").write_text("1 2 3 4\n")
    (bbox_dir / "frame2.txt").write_text("")

    captured = {}

    def fake_to_excel(self, path, index=False):
        captured["df"] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    dest_dir = tmp_path / "out"
    convert_to_standard_layout(tmp_path.as_posix(), dest_dir.as_posix())

    assert "df" in captured
    df = captured["df"]
    assert set(df["FRAME"]) == {"frame1.jpg", "frame2.jpg"}
    assert df.loc[df["FRAME"] == "frame1.jpg", "BOUNDING BOX"].iloc[0] == "1,2,3,4"
    assert df.loc[df["FRAME"] == "frame2.jpg", "BOUNDING BOX"].iloc[0] == ""
    assert df.loc[df["FRAME"] == "frame1.jpg", "CENTER"].iloc[0] == "C1"
    assert df["IMAGE TYPE"].eq("WL").all()
    assert df["POLYP ID"].isna().all()
    assert df.loc[df["FRAME"] == "frame1.jpg", "WIDTH"].iloc[0] == 10
    assert df.loc[df["FRAME"] == "frame2.jpg", "HEIGHT"].iloc[0] == 12


def test_convert_to_standard_layout_skips_excluded_frames(tmp_path, monkeypatch):
    images_dir = tmp_path / "data_C2" / "images_C2"
    bbox_dir = tmp_path / "data_C2" / "bbox_C2"
    images_dir.mkdir(parents=True)
    bbox_dir.mkdir(parents=True)

    # Excluded frame and a valid frame
    Image.new("RGB", (5, 5)).save(images_dir / "EndoCV2021_001010.jpg")
    Image.new("RGB", (6, 6)).save(images_dir / "EndoCV2021_001011.jpg")
    (bbox_dir / "EndoCV2021_001010.txt").write_text("")
    (bbox_dir / "EndoCV2021_001011.txt").write_text("1 1 2 2\n")

    captured = {}

    def fake_to_excel(self, path, index=False):
        captured["df"] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    dest_dir = tmp_path / "out"
    convert_to_standard_layout(tmp_path.as_posix(), dest_dir.as_posix())

    assert "df" in captured
    df = captured["df"]
    assert set(df["FRAME"]) == {"EndoCV2021_001011.jpg"}
