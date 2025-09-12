import pandas as pd

from polypdb.merge import relink_negative_cases


def test_relink_negative_cases(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()

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
    # existing cases.parquet containing positive case metadata
    pd.DataFrame(
        [
            {
                "global_case_id": "P1",
                "dataset": "sun",
                "center_id": None,
                "shape": "sessile",
                "size_mm": 5.0,
                "location": "rectum",
                "n_frames": 1,
                "source_lesion_id": "1",
            }
        ]
    ).to_parquet(manifests / "cases.parquet", index=False)

    relink_negative_cases(tmp_path)

    frames_df = pd.read_parquet(manifests / "frames.parquet")
    cases_df = pd.read_parquet(manifests / "cases.parquet")

    # positive frame untouched
    assert (
        frames_df.loc[frames_df["global_frame_id"] == "P1.000001", "global_case_id"].item()
        == "P1"
    )

    # negatives relinked
    neg_cases = frames_df[~frames_df["has_polyp"]]
    assert neg_cases["global_case_id"].unique().tolist() == ["NEG_SUN_g1"]
    assert neg_cases["dataset"].unique().tolist() == ["sun"]
    assert neg_cases["group"].unique().tolist() == ["g1"]

    # cases merged without losing positive metadata
    assert set(cases_df["global_case_id"]) == {"P1", "NEG_SUN_g1"}
    assert cases_df.loc[cases_df["global_case_id"] == "NEG_SUN_g1", "n_frames"].item() == 2
    assert cases_df.loc[cases_df["global_case_id"] == "P1", "n_frames"].item() == 1
    assert cases_df.loc[cases_df["global_case_id"] == "P1", "shape"].item() == "sessile"
