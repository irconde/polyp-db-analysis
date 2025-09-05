import pandas as pd
from pathlib import Path
from PIL import Image

from polypdb.merge import export_parquet_metadata


def test_export_parquet_includes_center_id(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()
    # Create a simple image
    Image.new("RGB", (10, 20)).save(images_dir / "frame1.jpg")

    df = pd.DataFrame([
        {
            "global_lesion_id": "L1",
            "global_frame_id": "L1.000001",
            "frame": "frame1.jpg",
            "dataset": "polypgen",
            "original_frame": "frame1.jpg",
            "n_frames": 1,
            "lesion_id": "1",
            "bounding_boxes": [],
            "morphology_shape": None,
            "morphology_size": None,
            "image_type": None,
            "morphology_location": None,
            "center_id": "C1",
        }
    ])

    export_parquet_metadata(df, canonical_base.as_posix())

    manifests = canonical_base / "manifests"
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    cases_df = pd.read_parquet(manifests / "cases.parquet")

    assert "center_id" in frames_df.columns
    assert frames_df.loc[0, "center_id"] == "C1"
    assert "center_id" in cases_df.columns
    assert cases_df.loc[0, "center_id"] == "C1"
