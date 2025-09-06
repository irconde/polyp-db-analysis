import pandas as pd
from PIL import Image

from polypdb.merge import export_binary_classification_dataset


def test_export_binary_classification_dataset(tmp_path):
    canonical = tmp_path / "canonical"
    images_dir = canonical / "images"
    manifests_dir = canonical / "manifests"
    images_dir.mkdir(parents=True)
    manifests_dir.mkdir()

    # Create sample positive and negative images
    Image.new("RGB", (10, 10)).save(images_dir / "pos.jpg")
    Image.new("RGB", (10, 10)).save(images_dir / "neg.jpg")

    df = pd.DataFrame([
        {"img_path": "images/pos.jpg", "has_polyp": True},
        {"img_path": "images/neg.jpg", "has_polyp": False},
    ])
    df.to_parquet(manifests_dir / "frames.parquet", index=False)

    out_dir = tmp_path / "flat"
    export_binary_classification_dataset(canonical, out_dir)

    assert (out_dir / "class_1" / "pos.jpg").exists()
    assert not (out_dir / "class_1" / "neg.jpg").exists()
    assert (out_dir / "class_0" / "neg.jpg").exists()
