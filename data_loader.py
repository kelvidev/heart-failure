import pandas as pd

import config


def load_dataset():
    df = pd.read_csv(config.DATASET_PATH)
    return df


def split_features_outcome(df):
    outcome = df[config.TARGET_COL].copy()
    features = df.drop(columns=[config.TARGET_COL]).copy()
    return features, outcome
