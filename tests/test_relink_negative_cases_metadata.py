import pandas as pd

from polypdb.merge import relink_negative_cases


def test_relink_negative_cases_preserves_metadata(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()

    # Prepare frames with one positive case and two negative frames needing relink
    frames = pd.DataFrame(
        [
            {
                "global_frame_id": "P1.000001",
                "global_case_id": "P1",
                "dataset": "sun",
                "has_polyp": True,
            },
            {
                "global_frame_id": "NEG_SUN_g1.000001",
                "global_case_id": "NEG_SUN_g1.000001",
                "dataset": None,
                "has_polyp": False,
            },
            {
                "global_frame_id": "NEG_SUN_g1.000002",
                "global_case_id": "NEG_SUN_g1.000002",
                "dataset": None,
                "has_polyp": False,
            },
        ]
    )
    frames.to_parquet(manifests / "frames.parquet", index=False)

    # Existing cases contain metadata for the positive case
    cases = pd.DataFrame(
        [
            {
                "global_case_id": "P1",
                "dataset": "sun",
                "center_id": "c1",
                "shape": "sessile",
                "size_mm": 5.0,
                "location": "rectum",
                "n_frames": 1,
                "source_lesion_id": "1",
            }
        ]
    )
    cases.to_parquet(manifests / "cases.parquet", index=False)

    relink_negative_cases(tmp_path)

    cases_df = pd.read_parquet(manifests / "cases.parquet")

    # Positive case metadata should be preserved
    pos = cases_df[cases_df["global_case_id"] == "P1"].iloc[0]
    assert pos["shape"] == "sessile"
    assert pos["size_mm"] == 5.0
    assert pos["location"] == "rectum"
    assert pos["source_lesion_id"] == "1"

    # Negative case rows should exist with metadata fields filled with NA
    neg = cases_df[cases_df["global_case_id"] == "NEG_SUN_g1"].iloc[0]
    assert neg["n_frames"] == 2
    assert neg["dataset"] == "sun"
    assert pd.isna(neg["shape"])
    assert pd.isna(neg["size_mm"])
    assert pd.isna(neg["location"])
    assert pd.isna(neg["source_lesion_id"])
