import pandas as pd
import numpy as np

from polypdb.preprocess.sun import _clip_bbox, _clip_bboxes


def test_piba_bbox_vectorization():
    df = pd.DataFrame(
        {
            "X": [10, 0],
            "Y": [5, 15],
            "BBOX WIDTH": [20, 30],
            "BBOX HEIGHT": [10, 5],
        }
    )

    expected = df.apply(
        lambda r: f"{r['X']},{r['Y']},{r['X'] + r['BBOX WIDTH']},{r['Y'] + r['BBOX HEIGHT']}",
        axis=1,
    )

    result = (
        df.assign(x2=df["X"] + df["BBOX WIDTH"], y2=df["Y"] + df["BBOX HEIGHT"])[
            ["X", "Y", "x2", "y2"]
        ]
        .astype(int)
        .astype(str)
        .agg(",".join, axis=1)
    )

    pd.testing.assert_series_equal(result, expected)


def test_polar_bbox_vectorization():
    df = pd.DataFrame(
        {
            "x": [5, 0],
            "y": [5, 10],
            "w": [10, 20],
            "h": [20, 5],
            "WIDTH": [20, 20],
            "HEIGHT": [20, 20],
        }
    )

    expected_w = df.apply(lambda r: min(r["w"], r["WIDTH"] - r["x"]), axis=1)
    expected_h = df.apply(lambda r: min(r["h"], r["HEIGHT"] - r["y"]), axis=1)
    expected_bbox = df.assign(w=expected_w, h=expected_h).apply(
        lambda r: f"{r['x']},{r['y']},{r['x'] + r['w']},{r['y'] + r['h']}",
        axis=1,
    )

    df_vec = df.copy()
    df_vec["w"] = np.minimum(df_vec["w"], df_vec["WIDTH"] - df_vec["x"])
    df_vec["h"] = np.minimum(df_vec["h"], df_vec["HEIGHT"] - df_vec["y"])
    result = (
        df_vec.assign(x2=df_vec["x"] + df_vec["w"], y2=df_vec["y"] + df_vec["h"])[
            ["x", "y", "x2", "y2"]
        ]
        .astype(int)
        .astype(str)
        .agg(",".join, axis=1)
    )

    pd.testing.assert_series_equal(result, expected_bbox)


def test_sun_bbox_clipping_vectorization():
    df = pd.DataFrame(
        {
            "BOUNDING BOX": ["-5,5,15,10 0,-10,10,10", ""],
            "WIDTH": [20, 10],
            "HEIGHT": [10, 5],
        }
    )

    expected = df.apply(
        lambda r: _clip_bbox(r["BOUNDING BOX"], r["WIDTH"], r["HEIGHT"]), axis=1
    )

    result = _clip_bboxes(df["BOUNDING BOX"], df["WIDTH"], df["HEIGHT"])

    pd.testing.assert_series_equal(result, expected)

