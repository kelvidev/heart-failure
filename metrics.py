import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

import config


def internal_metrics(X, labels):
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def cluster_sizes(labels):
    values, counts = np.unique(labels, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def cluster_profile(features, labels):
    df = features.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster").mean()
    return profile


def outcome_crosstab(outcome, labels):
    table = pd.crosstab(
        pd.Series(labels, name="cluster"),
        pd.Series(outcome.values, name=config.TARGET_COL),
    )
    return table
