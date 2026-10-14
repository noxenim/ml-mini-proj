"""
clustering.py - Customer segmentation on Mall_Customers dataset.
Implements K-Means, Agglomerative Clustering, and DBSCAN with hyperparameter evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clusters(X: np.ndarray, labels: np.ndarray, exclude_noise: bool = True) -> Dict[str, float]:
    """
    Calculate Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
    Handles noise points (-1) safely.
    """
    if exclude_noise:
        mask = labels != -1
        eval_X = X[mask]
        eval_labels = labels[mask]
    else:
        eval_X = X
        eval_labels = labels

    unique_clusters = set(eval_labels)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "valid": False
        }

    sil = float(silhouette_score(eval_X, eval_labels))
    db = float(davies_bouldin_score(eval_X, eval_labels))
    ch = float(calinski_harabasz_score(eval_X, eval_labels))

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "valid": True
    }


def run_clustering_experiments(df: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes clustering on Annual Income and Spending Score.
    Compares K-Means, Agglomerative, and DBSCAN.
    """
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. K-Means Parameter Sweep (k = 2 to 10)
    k_range = list(range(2, 11))
    kmeans_results = []
    inertias = []
    kmeans_silhouettes = []
    kmeans_models = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        metrics = evaluate_clusters(X_scaled, labels)
        inertias.append(float(km.inertia_))
        kmeans_silhouettes.append(metrics["silhouette"])
        kmeans_models[k] = (km, labels, metrics)
        kmeans_results.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"]
        })

    # Select best K-Means (highest silhouette score)
    best_k = max(k_range, key=lambda k: kmeans_models[k][2]["silhouette"])
    best_km, best_km_labels, best_km_metrics = kmeans_models[best_k]

    # 2. Agglomerative Clustering Evaluation (ward, complete, average)
    agg_configs = [
        {"n_clusters": 5, "linkage": "ward"},
        {"n_clusters": 5, "linkage": "complete"},
        {"n_clusters": 5, "linkage": "average"},
        {"n_clusters": 4, "linkage": "ward"},
        {"n_clusters": 6, "linkage": "ward"}
    ]
    agg_results = []
    best_agg_score = -1.0
    best_agg_model = None
    best_agg_labels = None
    best_agg_cfg = None
    best_agg_metrics = None

    for cfg in agg_configs:
        agg = AgglomerativeClustering(n_clusters=cfg["n_clusters"], linkage=cfg["linkage"])
        labels = agg.fit_predict(X_scaled)
        metrics = evaluate_clusters(X_scaled, labels)
        agg_results.append({
            **cfg,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"]
        })
        if metrics["silhouette"] > best_agg_score:
            best_agg_score = metrics["silhouette"]
            best_agg_model = agg
            best_agg_labels = labels
            best_agg_cfg = cfg
            best_agg_metrics = metrics

    # 3. DBSCAN Tuning (eps sweep across min_samples)
    dbscan_configs = [
        {"eps": 0.35, "min_samples": 5},
        {"eps": 0.38, "min_samples": 6},
        {"eps": 0.30, "min_samples": 5},
        {"eps": 0.40, "min_samples": 5},
        {"eps": 0.25, "min_samples": 4}
    ]
    dbscan_results = []
    best_dbscan_score = -1.0
    best_dbscan_model = None
    best_dbscan_labels = None
    best_dbscan_cfg = None
    best_dbscan_metrics = None
    best_dbscan_clusters = 0
    best_dbscan_noise = 0

    for cfg in dbscan_configs:
        db = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"])
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        metrics = evaluate_clusters(X_scaled, labels, exclude_noise=True)
        
        dbscan_results.append({
            **cfg,
            "clusters": n_clusters,
            "noise_points": n_noise,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"]
        })

        if metrics["valid"] and metrics["silhouette"] > best_dbscan_score:
            best_dbscan_score = metrics["silhouette"]
            best_dbscan_model = db
            best_dbscan_labels = labels
            best_dbscan_cfg = cfg
            best_dbscan_metrics = metrics
            best_dbscan_clusters = n_clusters
            best_dbscan_noise = n_noise

    # Build Comparison Summary Table
    comparison_data = [
        {
            "Algorithm": "K-Means",
            "Clusters": best_k,
            "Noise Points": 0,
            "Silhouette Score": round(best_km_metrics["silhouette"], 4),
            "Davies-Bouldin Index": round(best_km_metrics["davies_bouldin"], 4),
            "Calinski-Harabasz Score": round(best_km_metrics["calinski_harabasz"], 2),
            "Selected Configuration": f"k={best_k}, n_init=10, random_state={random_state}"
        },
        {
            "Algorithm": "Agglomerative (Hierarchical)",
            "Clusters": best_agg_cfg["n_clusters"],
            "Noise Points": 0,
            "Silhouette Score": round(best_agg_metrics["silhouette"], 4),
            "Davies-Bouldin Index": round(best_agg_metrics["davies_bouldin"], 4),
            "Calinski-Harabasz Score": round(best_agg_metrics["calinski_harabasz"], 2),
            "Selected Configuration": f"n_clusters={best_agg_cfg['n_clusters']}, linkage='{best_agg_cfg['linkage']}'"
        },
        {
            "Algorithm": "DBSCAN",
            "Clusters": best_dbscan_clusters,
            "Noise Points": best_dbscan_noise,
            "Silhouette Score": round(best_dbscan_metrics["silhouette"], 4),
            "Davies-Bouldin Index": round(best_dbscan_metrics["davies_bouldin"], 4),
            "Calinski-Harabasz Score": round(best_dbscan_metrics["calinski_harabasz"], 2),
            "Selected Configuration": f"eps={best_dbscan_cfg['eps']}, min_samples={best_dbscan_cfg['min_samples']} (Metrics on {len(X)-best_dbscan_noise} non-noise points)"
        }
    ]

    df_comparison = pd.DataFrame(comparison_data)

    clustering_artifacts = {
        "X_raw": X,
        "X_scaled": X_scaled,
        "features": features,
        "scaler": scaler,
        "kmeans_sweep": {
            "k_range": k_range,
            "inertias": inertias,
            "silhouettes": kmeans_silhouettes,
            "best_k": best_k
        },
        "best_kmeans": {
            "model": best_km,
            "labels": best_km_labels,
            "metrics": best_km_metrics
        },
        "best_agglomerative": {
            "model": best_agg_model,
            "labels": best_agg_labels,
            "metrics": best_agg_metrics,
            "config": best_agg_cfg
        },
        "best_dbscan": {
            "model": best_dbscan_model,
            "labels": best_dbscan_labels,
            "metrics": best_dbscan_metrics,
            "config": best_dbscan_cfg,
            "clusters": best_dbscan_clusters,
            "noise": best_dbscan_noise
        },
        "comparison_table": df_comparison
    }

    return df_comparison, clustering_artifacts
