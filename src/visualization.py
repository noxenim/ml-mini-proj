"""
visualization.py - Production-quality plotting routines using Matplotlib and Seaborn.
Generates readable, publication-grade figures with consistent styling, annotations, and units.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set global aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_clustering_visualizations(artifacts: Dict[str, Any], output_dir: str) -> List[str]:
    """Generates all clustering visual artifacts."""
    saved_files = []
    os.makedirs(output_dir, exist_ok=True)

    X = artifacts["X_raw"]
    feat_x, feat_y = artifacts["features"][0], artifacts["features"][1]

    # 1. Comparison of Cluster Assignments across Algorithms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    # Panel 1: K-Means
    km_labels = artifacts["best_kmeans"]["labels"]
    km_k = artifacts["kmeans_sweep"]["best_k"]
    palette_km = sns.color_palette("tab10", km_k)
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=km_labels, palette=palette_km,
        ax=axes[0], s=60, alpha=0.9, edgecolor="k", linewidth=0.5, legend="full"
    )
    # Centroids in original scale
    scaler = artifacts["scaler"]
    centroids_orig = scaler.inverse_transform(artifacts["best_kmeans"]["model"].cluster_centers_)
    axes[0].scatter(
        centroids_orig[:, 0], centroids_orig[:, 1],
        marker="X", s=220, color="crimson", edgecolor="black", linewidth=1.5,
        label="Centroids", zorder=5
    )
    axes[0].set_title(f"K-Means (k={km_k})\nSilhouette: {artifacts['best_kmeans']['metrics']['silhouette']:.3f}")
    axes[0].set_xlabel(feat_x)
    axes[0].set_ylabel(feat_y)
    axes[0].legend(title="Cluster", loc="upper right")

    # Panel 2: Agglomerative
    agg_labels = artifacts["best_agglomerative"]["labels"]
    agg_k = artifacts["best_agglomerative"]["config"]["n_clusters"]
    palette_agg = sns.color_palette("tab10", agg_k)
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=agg_labels, palette=palette_agg,
        ax=axes[1], s=60, alpha=0.9, edgecolor="k", linewidth=0.5, legend="full"
    )
    axes[1].set_title(f"Agglomerative (k={agg_k}, Ward)\nSilhouette: {artifacts['best_agglomerative']['metrics']['silhouette']:.3f}")
    axes[1].set_xlabel(feat_x)
    axes[1].legend(title="Cluster", loc="upper right")

    # Panel 3: DBSCAN
    db_labels = artifacts["best_dbscan"]["labels"]
    unique_db = sorted(list(set(db_labels)))
    db_colors = {}
    cluster_idx = 0
    palette_db = sns.color_palette("tab10", len(unique_db))
    for lbl in unique_db:
        if lbl == -1:
            db_colors[lbl] = (0.2, 0.2, 0.2, 0.5)  # Grey/black for noise
        else:
            db_colors[lbl] = palette_db[cluster_idx]
            cluster_idx += 1

    for lbl in unique_db:
        mask = (db_labels == lbl)
        if lbl == -1:
            axes[2].scatter(
                X[mask, 0], X[mask, 1], c="gray", marker="x", s=50,
                alpha=0.7, label=f"Noise ({mask.sum()} pts)", linewidth=1.5
            )
        else:
            axes[2].scatter(
                X[mask, 0], X[mask, 1], c=[db_colors[lbl]], s=60,
                alpha=0.9, edgecolor="k", linewidth=0.5, label=f"Cluster {lbl}"
            )
    axes[2].set_title(f"DBSCAN (eps={artifacts['best_dbscan']['config']['eps']}, min_s={artifacts['best_dbscan']['config']['min_samples']})\n"
                      f"{artifacts['best_dbscan']['clusters']} Clusters, {artifacts['best_dbscan']['noise']} Noise Points")
    axes[2].set_xlabel(feat_x)
    axes[2].legend(title="Groups", loc="upper right", fontsize=8)

    plt.suptitle("Mall Customer Segmentation: Model Comparison", y=1.02)
    path_clusters = os.path.join(output_dir, "clustering_clusters_comparison.png")
    plt.savefig(path_clusters)
    plt.close()
    saved_files.append(path_clusters)

    # 2. K-Means Elbow Curve & Silhouette Curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    k_vals = artifacts["kmeans_sweep"]["k_range"]
    inertias = artifacts["kmeans_sweep"]["inertias"]
    sils = artifacts["kmeans_sweep"]["silhouettes"]

    color_inertia = "royalblue"
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color_inertia)
    line1 = ax1.plot(k_vals, inertias, marker="o", color=color_inertia, linewidth=2, label="Inertia (Elbow)")
    ax1.tick_params(axis="y", labelcolor=color_inertia)
    ax1.axvline(x=artifacts["kmeans_sweep"]["best_k"], color="crimson", linestyle="--", alpha=0.7, label=f"Optimal k={artifacts['kmeans_sweep']['best_k']}")

    ax2 = ax1.twinx()
    color_sil = "darkorange"
    ax2.set_ylabel("Silhouette Score (Higher is Better)", color=color_sil)
    line2 = ax2.plot(k_vals, sils, marker="s", color=color_sil, linewidth=2, linestyle="-.", label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_sil)
    ax2.grid(False)

    lines = line1 + line2 + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title("K-Means: Elbow Method & Silhouette Score vs. Number of Clusters (k)")
    path_elbow = os.path.join(output_dir, "clustering_kmeans_elbow_silhouette.png")
    plt.savefig(path_elbow)
    plt.close()
    saved_files.append(path_elbow)

    # 3. Clustering Metrics Comparison Bar Chart
    df_comp = artifacts["comparison_table"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    sns.barplot(data=df_comp, x="Algorithm", y="Silhouette Score", hue="Algorithm", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title("Silhouette Score (Higher Better)")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=15)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")

    sns.barplot(data=df_comp, x="Algorithm", y="Davies-Bouldin Index", hue="Algorithm", ax=axes[1], palette="Reds_d", legend=False)
    axes[1].set_title("Davies-Bouldin Index (Lower Better)")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=15)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")

    sns.barplot(data=df_comp, x="Algorithm", y="Calinski-Harabasz Score", hue="Algorithm", ax=axes[2], palette="Greens_d", legend=False)
    axes[2].set_title("Calinski-Harabasz Score (Higher Better)")
    axes[2].set_ylabel("Score")
    axes[2].tick_params(axis="x", rotation=15)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")

    plt.suptitle("Clustering Evaluation Metrics Comparison", y=1.03)
    path_metrics = os.path.join(output_dir, "clustering_metrics_comparison.png")
    plt.savefig(path_metrics)
    plt.close()
    saved_files.append(path_metrics)

    return saved_files


def plot_classification_visualizations(df_results: pd.DataFrame, artifacts: Dict[str, Any], output_dir: str) -> List[str]:
    """Generates all classification visual artifacts."""
    saved_files = []
    os.makedirs(output_dir, exist_ok=True)

    # 1. Multi-metric Bar Chart Comparison
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    df_melt = pd.melt(df_results, id_vars=["Algorithm"], value_vars=metric_cols, var_name="Metric", value_name="Score")

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=df_melt, x="Algorithm", y="Score", hue="Metric", palette="viridis")
    plt.title("Classification Performance Comparison on Social Network Ads", pad=15)
    plt.ylabel("Score (0.0 to 1.0)")
    plt.ylim(0.5, 1.02)
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    path_perf = os.path.join(output_dir, "classification_metrics_comparison.png")
    plt.savefig(path_perf)
    plt.close()
    saved_files.append(path_perf)

    # 2. Confusion Matrices Grid (2x3)
    cms = artifacts["confusion_matrices"]
    model_names = list(cms.keys())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        cm = cms[name]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
            xticklabels=["Not Purchased (0)", "Purchased (1)"],
            yticklabels=["Not Purchased (0)", "Purchased (1)"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.suptitle("Confusion Matrices - Social Network Ads Classifiers", y=1.01)
    plt.tight_layout()
    path_cm = os.path.join(output_dir, "classification_confusion_matrices.png")
    plt.savefig(path_cm)
    plt.close()
    saved_files.append(path_cm)

    # 3. ROC Curves Comparison
    plt.figure(figsize=(8, 6.5))
    palette = sns.color_palette("tab10", len(model_names))

    for idx, name in enumerate(model_names):
        fpr, tpr, roc_auc = artifacts["roc_curves"][name]
        plt.plot(fpr, tpr, color=palette[idx], linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", linewidth=1.5, label="Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curves Comparison: Social Network Ads Models")
    plt.legend(loc="lower right", frameon=True)
    path_roc = os.path.join(output_dir, "classification_roc_curves.png")
    plt.savefig(path_roc)
    plt.close()
    saved_files.append(path_roc)

    return saved_files


def plot_boosting_visualizations(
    df_cls: pd.DataFrame,
    df_reg: pd.DataFrame,
    artifacts: Dict[str, Any],
    output_dir: str
) -> List[str]:
    """Generates all boosting visual artifacts."""
    saved_files = []
    os.makedirs(output_dir, exist_ok=True)
    cls_art = artifacts["classification"]

    # 1. Boosting Classification Performance Bar Chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    df_melt = pd.melt(df_cls, id_vars=["Algorithm"], value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_melt, x="Algorithm", y="Score", hue="Metric", palette="crest")
    plt.title("Boosting Algorithms: Smoker Status Classification", pad=15)
    plt.ylabel("Score")
    plt.ylim(0.85, 1.02)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    path_boost_metrics = os.path.join(output_dir, "boosting_metrics_comparison.png")
    plt.savefig(path_boost_metrics)
    plt.close()
    saved_files.append(path_boost_metrics)

    # 2. Confusion Matrices for Boosting Models
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    model_names = list(cls_art["confusion_matrices"].keys())

    for idx, name in enumerate(model_names):
        cm = cls_art["confusion_matrices"][name]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=axes[idx],
            xticklabels=["Non-Smoker (0)", "Smoker (1)"],
            yticklabels=["Non-Smoker (0)", "Smoker (1)"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices: Boosting Models (Smoker Prediction)", y=1.04)
    plt.tight_layout()
    path_boost_cm = os.path.join(output_dir, "boosting_confusion_matrices.png")
    plt.savefig(path_boost_cm)
    plt.close()
    saved_files.append(path_boost_cm)

    # 3. ROC Curves for Boosting Models
    plt.figure(figsize=(7.5, 6))
    palette = sns.color_palette("Set2", len(model_names))

    for idx, name in enumerate(model_names):
        fpr, tpr, roc_auc = cls_art["roc_curves"][name]
        plt.plot(fpr, tpr, color=palette[idx], linewidth=2.5, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.8, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Boosting Models (Smoker Prediction)")
    plt.legend(loc="lower right")
    path_boost_roc = os.path.join(output_dir, "boosting_roc_curves.png")
    plt.savefig(path_boost_roc)
    plt.close()
    saved_files.append(path_boost_roc)

    # 4. Feature Importances
    feat_imps = cls_art["feature_importances"]
    df_fi = pd.DataFrame(feat_imps).reset_index().rename(columns={"index": "Feature"})
    df_fi_melt = pd.melt(df_fi, id_vars=["Feature"], var_name="Algorithm", value_name="Importance")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_fi_melt, x="Feature", y="Importance", hue="Algorithm", palette="magma")
    plt.title("Feature Importance Across Boosting Algorithms (Predicting Smoker)")
    plt.ylabel("Relative Feature Importance")
    plt.xticks(rotation=15)
    plt.legend(title="Algorithm")
    path_fi = os.path.join(output_dir, "boosting_feature_importances.png")
    plt.savefig(path_fi)
    plt.close()
    saved_files.append(path_fi)

    # 5. Companion Regression: Actual vs. Predicted Charges
    reg_art = artifacts["regression"]
    y_test_r = reg_art["y_test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    reg_names = list(reg_art["predictions"].keys())

    for idx, name in enumerate(reg_names):
        y_pred_r = reg_art["predictions"][name]
        axes[idx].scatter(y_test_r, y_pred_r, alpha=0.6, edgecolors="k", linewidths=0.5, color="steelblue")
        min_val = min(y_test_r.min(), y_pred_r.min())
        max_val = max(y_test_r.max(), y_pred_r.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=1.5, label="Ideal Line (y = x)")
        r2_val = df_reg.loc[df_reg["Algorithm"] == name, "R2-Score"].values[0]
        axes[idx].set_title(f"{name}\n(R² = {r2_val:.4f})")
        axes[idx].set_xlabel("Actual Charges ($)")
        if idx == 0:
            axes[idx].set_ylabel("Predicted Charges ($)")
        axes[idx].legend(loc="upper left")

    plt.suptitle("Companion Analysis: Actual vs. Predicted Charges ($)", y=1.02)
    path_reg_scatter = os.path.join(output_dir, "boosting_regression_actual_vs_pred.png")
    plt.savefig(path_reg_scatter)
    plt.close()
    saved_files.append(path_reg_scatter)

    return saved_files
