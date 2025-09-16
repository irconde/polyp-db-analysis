import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

import polypdb.merge as merge

from polypdb.merge import (
    POLYPGEN_EXCLUDED_NEGATIVE_SEQUENCES,
    compute_file_sha1,
    copy_and_rename_negative_images,
    process_negative_images,
)
from polypdb.utils.image_ops import get_image_size


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


def test_polypgen_negative_sequence_exclusions(tmp_path):
    source_dir = tmp_path / "polypgen_negatives"

    for seq in POLYPGEN_EXCLUDED_NEGATIVE_SEQUENCES:
        seq_dir = source_dir / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), color="white").save(seq_dir / "frame1.jpg")

    dest_dir = tmp_path / "canonical" / "images"
    records, original_count = copy_and_rename_negative_images(
        str(source_dir), str(dest_dir), dataset_name="polypgen"
    )

    assert original_count == len(POLYPGEN_EXCLUDED_NEGATIVE_SEQUENCES)
    assert records == []
    assert dest_dir.exists()
    assert list(dest_dir.iterdir()) == []


def test_polypgen_negatives_preserve_source_files(tmp_path):
    source_dir = tmp_path / "polypgen_negatives"
    seq_dir = source_dir / "seq8_neg"
    seq_dir.mkdir(parents=True)

    original = Image.new("RGB", (10, 20), color="white")
    original.save(seq_dir / "frame1.jpg")

    dest_dir = tmp_path / "canonical" / "images"
    records, original_count = copy_and_rename_negative_images(
        str(source_dir), str(dest_dir), dataset_name="polypgen"
    )

    assert original_count == 1
    assert len(records) == 1

    src_size = get_image_size(seq_dir / "frame1.jpg")
    assert src_size == (10, 20)

    dest_name = Path(records[0]["img_path"]).name
    dest_path = dest_dir / dest_name
    assert dest_path.exists()
    assert get_image_size(dest_path) == (10, 20)

    process_negative_images(
        dest_dir, 32, 32, files=[dest_name]
    )

    # Original file should remain untouched while the canonical copy is resized
    assert get_image_size(seq_dir / "frame1.jpg") == (10, 20)
    assert get_image_size(dest_path) == (32, 32)

    # Metadata for the copied frame should be updated after sanitisation
    width, height = get_image_size(dest_path)
    records[0]["img_width"] = width
    records[0]["img_height"] = height
    records[0]["content_sha1"] = compute_file_sha1(dest_path)

    assert records[0]["img_width"] == 32
    assert records[0]["img_height"] == 32
    assert isinstance(records[0]["content_sha1"], str)
    assert len(records[0]["content_sha1"]) == 40


def test_process_negative_images_polypgen_sequence_uses_custom_crop(tmp_path, monkeypatch):
    img_path = tmp_path / "NEGSEQ.seq1_neg.00000.jpg"
    Image.new("RGB", (20, 20), color="white").save(img_path)

    calls = {}

    def fake_crop(img, **kwargs):
        calls["size"] = img.size
        return img.crop((1, 0, img.width, img.height))

    monkeypatch.setattr(merge, "crop_polypgen_negative_image", fake_crop)

    process_negative_images(tmp_path, 16, 16)

    assert "size" in calls
    with Image.open(img_path) as result:
        assert result.size == (16, 16)


def test_process_negative_images_regular_frames_skip_polypgen_crop(tmp_path, monkeypatch):
    img_path = tmp_path / "NEG_SUN_clipA.000001.jpg"
    Image.new("RGB", (18, 12), color="white").save(img_path)

    def fail_crop(*args, **kwargs):
        raise AssertionError("crop called")

    monkeypatch.setattr(merge, "crop_polypgen_negative_image", fail_crop)

    process_negative_images(tmp_path, 16, 16)

    with Image.open(img_path) as result:
        assert result.size == (16, 16)
