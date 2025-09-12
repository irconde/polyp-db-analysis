import pandas as pd
from PIL import Image

from polypdb.merge import rebuild_parquet_manifests


def test_rebuild_parquet_manifests(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()

    # create image
    Image.new("RGB", (10, 20)).save(images_dir / "frame1.jpg")

    df = pd.DataFrame(
        {
            "ORIGINAL DATASET": ["SUN"],
            "POLYP ID": ["SUN_0001"],
            "lesion_id": ["0001"],
            "global_lesion_id": ["SUN_0001"],
            "FRAME": ["frame1.jpg"],
            "ORIGINAL FRAME": ["frame1.jpg"],
            "BOUNDING BOX": ["1,2,3,4"],
            "SHAPE": ["Is"],
            "SIZE": [5],
            "LOCATION": ["colon"],
            "IMAGE TYPE": ["WL"],
        }
    )

    metadata_path = canonical_base / "metadata.xlsx"
    df.to_excel(metadata_path, index=False)

    rebuild_parquet_manifests(
        metadata_path.as_posix(), canonical_base.as_posix(), images_dir.as_posix()
    )

    manifests = canonical_base / "manifests"
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    cases_df = pd.read_parquet(manifests / "cases.parquet")

    assert len(frames_df) == 1
    assert len(cases_df) == 1
    assert frames_df.loc[0, "global_case_id"] == "SUN_0001"
    assert cases_df.loc[0, "global_case_id"] == "SUN_0001"
