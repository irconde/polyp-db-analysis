import pandas as pd
from PIL import Image
from polypdb.preprocess.sun import convert_to_standard_layout

def test_convert_to_standard_layout_parses_size_mm(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("ID,LOCATION,SHAPE,SIZE\n1,colon,Is,5mm\n")
    ann_dir = tmp_path / "sundatabase_positive" / "annotation_txt"
    ann_dir.mkdir(parents=True)
    (ann_dir / "case1.txt").write_text("img1.png 1,2,3,4,0\n")
    case_dir = tmp_path / "sundatabase_positive" / "case1"
    case_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(case_dir / "img1.png")

    captured = {}
    def fake_to_excel(self, path, index=False):
        captured['df'] = self.copy()
    monkeypatch.setattr(pd.DataFrame, 'to_excel', fake_to_excel)

    dest_dir = tmp_path / "out"
    convert_to_standard_layout(tmp_path.as_posix(), dest_dir.as_posix())

    assert 'df' in captured
    assert captured['df']['SIZE'].iloc[0] == 5

