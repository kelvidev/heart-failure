import warnings

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import config
import data_loader
import preprocessing
import metrics
import persistence

warnings.filterwarnings("ignore")


def candidate_labelings(X, k):
    return {
        "KMeans": KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(X),
        "Agglomerative": AgglomerativeClustering(n_clusters=k).fit_predict(X),
        "GaussianMixture": GaussianMixture(n_components=k, random_state=config.RANDOM_STATE).fit_predict(X),
    }


def compare_candidates(X):
    table = {}
    for k in config.K_RANGE:
        labelings = candidate_labelings(X, k)
        for name, labels in labelings.items():
            table[(name, k)] = silhouette_score(X, labels)
    return table


def best_kmeans_k(X):
    best_k = None
    best_score = None
    for k in config.K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(X)
        score = silhouette_score(X, labels)
        if best_score is None or score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def run_training():
    df = data_loader.load_dataset()
    features, outcome = data_loader.split_features_outcome(df)

    X_df, scaler = preprocessing.fit_transform(features)
    X = X_df.values

    print("Comparacao de metaestimadores (metrica = silhouette, maior e melhor):")
    table = compare_candidates(X)
    for k in config.K_RANGE:
        row = "  k={}  ".format(k)
        for name in ["KMeans", "Agglomerative", "GaussianMixture"]:
            row += "{}={:.4f}  ".format(name, table[(name, k)])
        print(row)

    k, sil = best_kmeans_k(X)
    print("\nMetaestimador escolhido: KMeans (suporta inferencia out-of-sample via centroides)")
    print("Melhor k para KMeans por silhouette: k={} (silhouette={:.4f})".format(k, sil))

    model = KMeans(n_clusters=k, random_state=config.RANDOM_STATE)
    labels = model.fit_predict(X)

    scores = metrics.internal_metrics(X, labels)
    print("\nMetricas internas do modelo final:")
    print("  silhouette        = {:.4f}".format(scores["silhouette"]))
    print("  davies_bouldin    = {:.4f}  (menor e melhor)".format(scores["davies_bouldin"]))
    print("  calinski_harabasz = {:.2f}  (maior e melhor)".format(scores["calinski_harabasz"]))

    print("\nTamanho dos clusters:", metrics.cluster_sizes(labels))

    print("\nPerfil medio de cada cluster (variaveis no espaco original):")
    print(metrics.cluster_profile(features, labels).round(2).to_string())

    print("\nValidacao externa (cruzamento com DEATH_EVENT, nao usado no treino):")
    print(metrics.outcome_crosstab(outcome, labels).to_string())

    metadata = {
        "model_name": "KMeans",
        "n_clusters": k,
        "continuous_cols": config.CONTINUOUS_COLS,
        "binary_cols": config.BINARY_COLS,
        "feature_order": list(features.columns),
        "internal_metrics": scores,
        "cluster_sizes": metrics.cluster_sizes(labels),
    }

    persistence.save_pickle(model, config.MODEL_PATH)
    persistence.save_pickle(scaler, config.SCALER_PATH)
    persistence.save_pickle(metadata, config.META_PATH)
    print("\nArtefatos salvos em:", config.MODELS_DIR)
    return model, scaler, metadata


if __name__ == "__main__":
    run_training()
