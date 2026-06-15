import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def fit_scaler(features):
    scaler = StandardScaler()
    scaler.fit(features[config.CONTINUOUS_COLS])
    return scaler


def transform_features(features, scaler):
    continuous = pd.DataFrame(
        scaler.transform(features[config.CONTINUOUS_COLS]),
        columns=config.CONTINUOUS_COLS,
        index=features.index,
    )
    binary = features[config.BINARY_COLS].astype(int).copy()
    binary.index = features.index
    joined = pd.concat([continuous, binary], axis=1)
    return joined


def fit_transform(features):
    scaler = fit_scaler(features)
    transformed = transform_features(features, scaler)
    return transformed, scaler
