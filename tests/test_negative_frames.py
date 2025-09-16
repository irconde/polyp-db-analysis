import os

import pandas as pd
from PIL import Image, ImageDraw

from polypdb.merge import copy_and_rename_negative_images


def test_copy_and_rename_negative_images_with_explicit_dataset(tmp_path):
    # create structure: <source_dir>/<group>/image
    source_dir = tmp_path / "sun_negatives"
    group_dir = source_dir / "clipA"
    group_dir.mkdir(parents=True)
    img_path = group_dir / "frame1.png"
    Image.new("RGB", (20, 30)).save(img_path)

    dest_dir = tmp_path / "canonical" / "images"
    records, original = copy_and_rename_negative_images(
        str(source_dir), str(dest_dir), dataset_name="sun"
    )

    expected_name = "NEG_SUN_clipA.000001.png"
    assert (dest_dir / expected_name).exists()
    assert original == 1
    assert len(records) == 1
    rec = records[0]
    assert rec["global_frame_id"] == "NEG_SUN_clipA.000001"
    assert rec["global_case_id"] == "NEG_SUN_clipA"
    assert rec["dataset"] == "sun"
    assert rec["source_frame_id"] == "frame1.png"
    assert rec["img_width"] == 20
    assert rec["img_height"] == 30
    assert rec["filename_ext"] == ".png"
    assert rec["img_path"] == os.path.join("images", expected_name)
    assert rec["modality"] is None
    assert rec["origin"] == "frame"
    assert pd.isna(rec["sequence_id"])
    assert pd.isna(rec["sequence_frame_index"])
    assert rec["center_known"] is False
    assert isinstance(rec["content_sha1"], str) and len(rec["content_sha1"]) == 40


def test_copy_and_rename_negative_images_without_groups(tmp_path):
    source_dir = tmp_path / "polar_negatives"
    source_dir.mkdir()
    img_path = source_dir / "neg1.jpg"
    Image.new("RGB", (10, 15)).save(img_path)

    dest_dir = tmp_path / "canonical" / "images"
    records, original = copy_and_rename_negative_images(
        str(source_dir), str(dest_dir), dataset_name="polar"
    )

    expected_name = "NEG_POLAR_polar_negatives.000001.jpg"
    assert (dest_dir / expected_name).exists()
    assert original == 1
    assert len(records) == 1
    rec = records[0]
    assert rec["global_frame_id"] == "NEG_POLAR_polar_negatives.000001"
    assert rec["global_case_id"] == "NEG_POLAR_polar_negatives"
    assert rec["dataset"] == "polar"
    assert rec["source_frame_id"] == "neg1.jpg"
    assert rec["img_width"] == 10
    assert rec["img_height"] == 15
    assert rec["filename_ext"] == ".jpg"
    assert rec["img_path"] == os.path.join("images", expected_name)
    assert rec["modality"] is None
    assert rec["origin"] == "frame"
    assert pd.isna(rec["sequence_id"])
    assert pd.isna(rec["sequence_frame_index"])
    assert rec["center_known"] is False
    assert isinstance(rec["content_sha1"], str) and len(rec["content_sha1"]) == 40


def test_copy_and_rename_negative_sequences(tmp_path):
    source_dir = tmp_path / "polypgen_negatives"
    seq_dir = source_dir / "seq1_neg"
    seq_dir.mkdir(parents=True)
    img1 = Image.new("RGB", (12, 18), color="white")
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle((0, 0, 5, 17), fill="red")
    img1.save(seq_dir / "frame1.jpg")
    img2 = Image.new("RGB", (12, 18), color="white")
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle((6, 0, 11, 17), fill="blue")
    img2.save(seq_dir / "frame2.jpg")

    dest_dir = tmp_path / "canonical" / "images"
    records, original = copy_and_rename_negative_images(
        str(source_dir), str(dest_dir), dataset_name="polypgen"
    )

    assert original == 2
    assert len(records) == 2
    first = records[0]
    second = records[1]

    assert first["global_frame_id"] == "NEGSEQ.seq1_neg.00000"
    assert second["global_frame_id"] == "NEGSEQ.seq1_neg.00001"
    assert first["global_case_id"] == "NEGSEQ_SEQ1_NEG"
    assert first["origin"] == "neg_seq"
    assert second["origin"] == "neg_seq"
    assert first["sequence_id"] == "seq1_neg"
    assert second["sequence_id"] == "seq1_neg"
    assert first["sequence_frame_index"] == 0
    assert second["sequence_frame_index"] == 1
    assert not first["center_known"]
    assert not second["center_known"]
    assert len(first["content_sha1"]) == 40
    assert len(second["content_sha1"]) == 40
    assert (dest_dir / "NEGSEQ.seq1_neg.00000.jpg").exists()
    assert (dest_dir / "NEGSEQ.seq1_neg.00001.jpg").exists()
