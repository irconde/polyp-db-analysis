import shutil

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
    assert frames_df.loc[0, "origin"] == "frame"
    assert pd.isna(frames_df.loc[0, "sequence_id"])
    assert pd.isna(frames_df.loc[0, "sequence_frame_index"])
    assert bool(frames_df.loc[0, "center_known"])
    assert isinstance(frames_df.loc[0, "content_sha1"], str)
    assert len(frames_df.loc[0, "content_sha1"]) == 40
    assert "center_id" in cases_df.columns
    assert cases_df.loc[0, "center_id"] == "C1"


def test_export_parquet_writes_annotations(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()
    Image.new("RGB", (20, 30)).save(images_dir / "frame1.jpg")

    df = pd.DataFrame(
        [
            {
                "global_lesion_id": "L1",
                "global_frame_id": "L1.000001",
                "frame": "frame1.jpg",
                "dataset": "polypgen",
                "original_frame": "frame1.jpg",
                "n_frames": 1,
                "lesion_id": "1",
                "bounding_boxes": [[1, 2, 6, 12], [0, 0, 5, 5]],
                "morphology_shape": None,
                "morphology_size": None,
                "image_type": None,
                "morphology_location": None,
                "center_id": "C1",
            }
        ]
    )

    export_parquet_metadata(df, canonical_base.as_posix())

    manifests = canonical_base / "manifests"
    ann_path = manifests / "annotations.parquet"
    assert ann_path.exists()

    ann_df = pd.read_parquet(ann_path).sort_values("ann_id").reset_index(drop=True)
    expected_ids = ["L1.000001.00", "L1.000001.01"]
    assert ann_df["ann_id"].tolist() == expected_ids
    assert set(ann_df.columns) == {
        "ann_id",
        "global_frame_id",
        "global_case_id",
        "center_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "bbox_format",
        "bbox_normalized",
    }
    assert ann_df.loc[0, "global_case_id"] == "L1"
    assert ann_df.loc[0, "bbox_x"] == 1.0
    assert ann_df.loc[0, "bbox_w"] == 5.0
    assert not ann_df["bbox_normalized"].any()


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
    assert not bool(frames_df.loc[0, "center_known"])


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
    Image.new("RGB", (10, 20), color="red").save(images_dir / "neg1.jpg")
    Image.new("RGB", (10, 20), color="blue").save(images_dir / "neg2.jpg")

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

    retained = export_parquet_metadata(df, canonical_base.as_posix(), negative_frames)

    manifests = canonical_base / "manifests"
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    cases_df = pd.read_parquet(manifests / "cases.parquet")
    assert len(frames_df) == 2
    assert len(cases_df) == 1
    assert cases_df.loc[0, "global_case_id"] == "NEG_SUN_g1"
    assert cases_df.loc[0, "n_frames"] == 2
    assert all(frames_df["origin"] == "frame")
    assert not frames_df["center_known"].any()
    assert frames_df["content_sha1"].str.len().eq(40).all()
    assert len(retained) == 2


def test_export_parquet_respects_write_cases_flag(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()
    Image.new("RGB", (10, 20)).save(images_dir / "frame1.jpg")
    Image.new("RGB", (10, 20), color="red").save(images_dir / "neg1.jpg")

    df = pd.DataFrame(
        [
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
        ]
    )

    negative_frames = [
        {
            "global_frame_id": "NEG_POLYPGEN_seq1.000001",
            "global_case_id": "NEG_POLYPGEN_seq1",
            "dataset": "polypgen",
            "source_frame_id": "neg1.jpg",
            "img_width": 10,
            "img_height": 20,
            "filename_ext": ".jpg",
            "img_path": "images/neg1.jpg",
            "modality": None,
        }
    ]

    retained = export_parquet_metadata(
        df, canonical_base.as_posix(), negative_frames, write_cases=False
    )

    manifests = canonical_base / "manifests"
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    assert len(frames_df) == 2
    assert len(retained) == 1
    assert not (manifests / "cases.parquet").exists()


def test_export_parquet_deduplicates_content_sha1(tmp_path):
    canonical_base = tmp_path
    images_dir = canonical_base / "images"
    images_dir.mkdir()
    Image.new("RGB", (10, 10), color="white").save(images_dir / "frame1.jpg")
    shutil.copy(images_dir / "frame1.jpg", images_dir / "frame2.jpg")

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
        },
        {
            "global_lesion_id": "L2",
            "global_frame_id": "L2.000001",
            "frame": "frame2.jpg",
            "dataset": "polypgen",
            "original_frame": "frame2.jpg",
            "n_frames": 1,
            "lesion_id": "2",
            "bounding_boxes": [],
            "morphology_shape": None,
            "morphology_size": None,
            "image_type": None,
            "morphology_location": None,
            "center_id": "C2",
        },
    ])

    export_parquet_metadata(df, canonical_base.as_posix())

    manifests = canonical_base / "manifests"
    frames_df = pd.read_parquet(manifests / "frames.parquet")
    cases_df = pd.read_parquet(manifests / "cases.parquet")

    assert len(frames_df) == 1
    assert frames_df.iloc[0]["global_frame_id"] == "L1.000001"
    assert cases_df["global_case_id"].tolist() == ["L1"]
    assert (images_dir / "frame2.jpg").exists() is False
