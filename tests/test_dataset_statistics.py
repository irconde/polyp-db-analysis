import pandas as pd

from polypdb.merge import dataset_statistics


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
