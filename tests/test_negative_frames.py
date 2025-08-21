import os
from PIL import Image

from polypdb.merge import copy_and_rename_negative_images


def test_copy_and_rename_negative_images_with_explicit_dataset(tmp_path):
    # create structure: <source_dir>/<group>/image
    source_dir = tmp_path / "sun_negatives"
    group_dir = source_dir / "clipA"
    group_dir.mkdir(parents=True)
    img_path = group_dir / "frame1.png"
    Image.new("RGB", (20, 30)).save(img_path)

    dest_dir = tmp_path / "canonical" / "images"
    records = copy_and_rename_negative_images(
        str(source_dir), str(dest_dir), dataset_name="sun"
    )

    expected_name = "NEG_SUN_clipA.000001.png"
    assert (dest_dir / expected_name).exists()
    assert records == [
        {
            "global_frame_id": "NEG_SUN_clipA.000001",
            "global_lesion_id": None,
            "dataset": "sun",
            "source_frame_id": "frame1.png",
            "img_width": 20,
            "img_height": 30,
            "filename_ext": ".png",
            "canonical_path": os.path.join(str(dest_dir), expected_name),
            "raw_path": str(img_path),
            "modality": None,
        }
    ]


def test_copy_and_rename_negative_images_without_groups(tmp_path):
    source_dir = tmp_path / "polar_negatives"
    source_dir.mkdir()
    img_path = source_dir / "neg1.jpg"
    Image.new("RGB", (10, 15)).save(img_path)

    dest_dir = tmp_path / "canonical" / "images"
    records = copy_and_rename_negative_images(
        str(source_dir), str(dest_dir), dataset_name="polar"
    )

    expected_name = "NEG_POLAR_polar_negatives.000001.jpg"
    assert (dest_dir / expected_name).exists()
    assert records == [
        {
            "global_frame_id": "NEG_POLAR_polar_negatives.000001",
            "global_lesion_id": None,
            "dataset": "polar",
            "source_frame_id": "neg1.jpg",
            "img_width": 10,
            "img_height": 15,
            "filename_ext": ".jpg",
            "canonical_path": os.path.join(str(dest_dir), expected_name),
            "raw_path": str(img_path),
            "modality": None,
        }
    ]
