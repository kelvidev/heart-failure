import numpy as np
import pandas as pd

import config
import preprocessing
import persistence


def load_artifacts():
    model = persistence.load_pickle(config.MODEL_PATH)
    scaler = persistence.load_pickle(config.SCALER_PATH)
    metadata = persistence.load_pickle(config.META_PATH)
    return model, scaler, metadata


def predict(samples, artifacts=None):
    if artifacts is None:
        artifacts = load_artifacts()
    model, scaler, metadata = artifacts

    df = pd.DataFrame(samples)
    df = df[metadata["feature_order"]]

    X_df = preprocessing.transform_features(df, scaler)
    X = X_df.values

    clusters = model.predict(X)
    distances = model.transform(X)
    nearest = distances.min(axis=1)
    return clusters, nearest


def demo():
    artifacts = load_artifacts()
    _, _, metadata = artifacts

    unknown_patients = [
        {
            "age": 75.0,
            "anaemia": 0,
            "creatinine_phosphokinase": 582,
            "diabetes": 0,
            "ejection_fraction": 20,
            "high_blood_pressure": 1,
            "platelets": 265000.0,
            "serum_creatinine": 1.9,
            "serum_sodium": 130,
            "sex": 1,
            "smoking": 0,
            "time": 4,
        },
        {
            "age": 50.0,
            "anaemia": 1,
            "creatinine_phosphokinase": 111,
            "diabetes": 0,
            "ejection_fraction": 45,
            "high_blood_pressure": 0,
            "platelets": 210000.0,
            "serum_creatinine": 1.0,
            "serum_sodium": 137,
            "sex": 0,
            "smoking": 0,
            "time": 200,
        },
    ]

    clusters, nearest = predict(unknown_patients, artifacts)

    print("Modelo: {} com {} clusters".format(metadata["model_name"], metadata["n_clusters"]))
    print()
    for i, (c, d) in enumerate(zip(clusters, nearest)):
        print("Paciente desconhecido #{}".format(i + 1))
        print("  grupo atribuido: cluster {}".format(int(c)))
        print("  distancia ao centroide mais proximo: {:.4f}".format(d))
        print()


if __name__ == "__main__":
    demo()
