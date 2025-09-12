import pandas as pd
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


def test_export_parquet_skips_cases_without_metadata(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()
    Image.new("RGB", (10, 20)).save(images_dir / "frame1.jpg")

    df = pd.DataFrame(
        [
            {
                "global_frame_id": "F1",
                "frame": "frame1.jpg",
                "dataset": "polypgen",
                "original_frame": "frame1.jpg",
                "bounding_boxes": [],
            }
        ]
    )

    export_parquet_metadata(df, canonical_base.as_posix())

    manifests = canonical_base / "manifests"
    assert not (manifests / "cases.parquet").exists()
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    assert frames_df.loc[0, "global_case_id"] == "F1"


def test_export_parquet_skips_cases_when_ids_empty(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()
    Image.new("RGB", (10, 20)).save(images_dir / "frame1.jpg")

    df = pd.DataFrame(
        [
            {
                "global_lesion_id": None,
                "lesion_id": None,
                "n_frames": 1,
                "global_frame_id": "F1",
                "frame": "frame1.jpg",
                "dataset": "polypgen",
                "original_frame": "frame1.jpg",
                "bounding_boxes": [],
            }
        ]
    )

    export_parquet_metadata(df, canonical_base.as_posix())

    manifests = canonical_base / "manifests"
    assert not (manifests / "cases.parquet").exists()
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    assert frames_df.loc[0, "global_case_id"] == "F1"


def test_export_parquet_records_negative_cases(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()
    Image.new("RGB", (10, 20)).save(images_dir / "neg1.jpg")
    Image.new("RGB", (10, 20)).save(images_dir / "neg2.jpg")

    df = pd.DataFrame([])
    negative_frames = [
        {
            "global_frame_id": "NEG_SUN_g1.000001",
            "global_case_id": "NEG_SUN_g1",
            "dataset": "sun",
            "source_frame_id": "neg1.jpg",
            "img_width": 10,
            "img_height": 20,
            "filename_ext": ".jpg",
            "img_path": "images/neg1.jpg",
            "modality": None,
        },
        {
            "global_frame_id": "NEG_SUN_g1.000002",
            "global_case_id": "NEG_SUN_g1",
            "dataset": "sun",
            "source_frame_id": "neg2.jpg",
            "img_width": 10,
            "img_height": 20,
            "filename_ext": ".jpg",
            "img_path": "images/neg2.jpg",
            "modality": None,
        },
    ]

    export_parquet_metadata(df, canonical_base.as_posix(), negative_frames)

    manifests = canonical_base / "manifests"
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    cases_df = pd.read_parquet(manifests / "cases.parquet")
    assert len(frames_df) == 2
    assert len(cases_df) == 1
    assert cases_df.loc[0, "global_case_id"] == "NEG_SUN_g1"
    assert cases_df.loc[0, "n_frames"] == 2
