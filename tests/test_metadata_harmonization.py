import pandas as pd

from polypdb.merge import harmonize_metadata, generate_global_lesion_ids


def test_harmonize_metadata_basic(tmp_path):
    # Prepare minimal dataframe emulating merged annotations
    df = pd.DataFrame(
        {
            "ORIGINAL DATASET": ["sun", "polar"],
            "POLYP ID": ["1", "2"],
            "FRAME": ["sun_a.png", "polar_a.png"],
            "ORIGINAL FRAME": ["sun_a.png", "polar_a.png"],
            "BOUNDING BOX": ["1,2,3,4", "5,6,7,8"],
            "SHAPE": ["Is", "Ip"],
            "SIZE": [5, 10],
            "LOCATION": ["colon", "rectum"],
            "IMAGE TYPE": ["WL", "NBI"],
        }
    )

    # Derive identifiers as performed in merge_datasets
    df["lesion_id"] = df["POLYP ID"].astype(str)
    df["global_lesion_id"] = generate_global_lesion_ids(df)
    df["POLYP ID"] = df["global_lesion_id"]

    image_dir = tmp_path.as_posix()
    harmonised = harmonize_metadata(df, image_dir)

    expected_cols = {
        "global_lesion_id",
        "dataset",
        "lesion_id",
        "n_frames",
        "frame",
        "original_frame",
        "image_paths",
        "bounding_boxes",
        "morphology_shape",
        "morphology_size",
        "image_type",
        "morphology_location",
    }

    assert expected_cols.issubset(harmonised.columns)
    # Preserve original morphology labels verbatim
    assert harmonised.loc[0, "morphology_shape"] == "Is"
    assert harmonised.loc[1, "morphology_shape"] == "Ip"
    assert harmonised.loc[0, "image_paths"] == [f"{image_dir}/sun_a.png"]
    assert harmonised.loc[0, "bounding_boxes"] == [[1, 2, 3, 4]]
    assert harmonised["n_frames"].tolist() == [1, 1]
