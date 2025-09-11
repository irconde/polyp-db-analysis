import pandas as pd

from polypdb.ultralytics import split_data


def test_split_data_with_counts():
    df = pd.DataFrame({
        'FRAME': [f"{i}.jpg" for i in range(10)],
        'POLYP ID': list(range(10)),
        'SHAPE': ['A'] * 5 + ['B'] * 5,
    })
    config = {
        'splitting_options': {
            'train_count': 4,
            'validation_count': 3,
            'split_type': 'RANDOM',
            'split_level': 'FRAME',
        }
    }
    train_df, val_df, test_df = split_data(df, config)
    assert len(train_df) == 4
    assert len(val_df) == 3
    assert len(test_df) == 3
