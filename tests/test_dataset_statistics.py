import pandas as pd

from polypdb.merge import dataset_statistics, dataset_statistics_from_canonical


def test_dataset_statistics():
    df = pd.DataFrame(
        {
            "global_lesion_id": ["A1", "A1", "B1"],
            "dataset": ["sun", "sun", "polar"],
            "morphology_shape": ["sessile", "sessile", "flat"],
        }
    )
    negative_records = [
        {"global_frame_id": "NEG_SUN_g1.000001", "dataset": "sun"},
        {"global_frame_id": "NEG_SUN_g1.000002", "dataset": "sun"},
        {"global_frame_id": "NEG_POLAR_g2.000001", "dataset": "polar"},
    ]
    stats = dataset_statistics(df, negative_records, negative_original_count=4)
    assert stats["original_total_samples"] == 7
    assert stats["total_samples"] == 6
    assert stats["case_distribution"] == {"positive": 2, "negative": 2}
    assert stats["morphology_distribution"] == {"sessile": 1, "flat": 1}
    assert stats["dataset_distribution"] == {"sun": 2, "polar": 2}
    assert stats["frames_per_case_summary"]["max"] == 2


def test_dataset_statistics_from_canonical(tmp_path):
    canonical = tmp_path / "canonical"
    manifests = canonical / "manifests"
    manifests.mkdir(parents=True)

    frames_df = pd.DataFrame(
        [
            {"global_frame_id": "A1.000001", "global_lesion_id": "A1", "dataset": "sun", "has_polyp": True},
            {"global_frame_id": "A1.000002", "global_lesion_id": "A1", "dataset": "sun", "has_polyp": True},
            {"global_frame_id": "B1.000001", "global_lesion_id": "B1", "dataset": "polar", "has_polyp": True},
            {"global_frame_id": "NEG_SUN_g1.000001", "global_lesion_id": None, "dataset": "sun", "has_polyp": False},
            {"global_frame_id": "NEG_SUN_g1.000002", "global_lesion_id": None, "dataset": "sun", "has_polyp": False},
            {"global_frame_id": "NEG_POLAR_g2.000001", "global_lesion_id": None, "dataset": "polar", "has_polyp": False},
        ]
    )
    frames_df.to_parquet(manifests / "frames.parquet", index=False)

    lesions_df = pd.DataFrame(
        [
            {"global_lesion_id": "A1", "dataset": "sun", "shape": "sessile", "n_frames": 2},
            {"global_lesion_id": "B1", "dataset": "polar", "shape": "flat", "n_frames": 1},
        ]
    )
    lesions_df.to_parquet(manifests / "lesions.parquet", index=False)

    stats = dataset_statistics_from_canonical(str(canonical))
    assert stats["original_total_samples"] == 6
    assert stats["total_samples"] == 6
    assert stats["case_distribution"] == {"positive": 2, "negative": 2}
    assert stats["morphology_distribution"] == {"sessile": 1, "flat": 1}
    assert stats["dataset_distribution"] == {"sun": 2, "polar": 2}
    assert stats["frames_per_case_summary"]["max"] == 2
