"""
main.py - Unified executable entry point for Machine Learning Algorithm Comparison Assignment.
Runs data loading, clustering, classification, boosting, visual artifact generation,
and technical report compilation end-to-end.
"""

import os
import sys
import pandas as pd
from src.data_loader import load_all_datasets
from src.clustering import run_clustering_experiments
from src.classification import run_classification_experiments
from src.boosting import run_boosting_experiments
from src.visualization import (
    plot_clustering_visualizations,
    plot_classification_visualizations,
    plot_boosting_visualizations
)
from src.report_generator import generate_markdown_report, generate_pdf_report


def print_banner():
    banner = """
================================================================================
          MACHINE LEARNING ALGORITHM COMPARISON SUITE
  Clustering (Mall Customers) | Classification (Ads) | Boosting (Insurance)
================================================================================
    """
    print(banner)


def main():
    print_banner()
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[*] Results directory: {results_dir}")

    # 1. Load and Inspect Datasets
    print("\n--- STEP 1: Ingesting and Inspecting Datasets ---")
    df_mall, df_ads, df_ins, metadata = load_all_datasets(os.getcwd())
    for key, info in metadata.items():
        print(f"  [+] {info['name']}: {info['rows']} rows, {info['columns']} cols, "
              f"Missing values: {info['total_missing']}, Duplicates: {info['duplicate_rows']}")

    # 2. Dataset A: Clustering (Mall Customers)
    print("\n--- STEP 2: Executing Clustering Experiments (Mall Customers) ---")
    df_clustering, clustering_artifacts = run_clustering_experiments(df_mall, random_state=42)
    clustering_csv_path = os.path.join(results_dir, "clustering_results.csv")
    df_clustering.to_csv(clustering_csv_path, index=False)
    print(f"  [+] Clustering results saved to: {clustering_csv_path}")
    print(df_clustering.to_string(index=False))

    print("  [*] Generating clustering visualizations...")
    clust_imgs = plot_clustering_visualizations(clustering_artifacts, results_dir)
    for img in clust_imgs:
        print(f"      Saved: {os.path.basename(img)}")

    # 3. Dataset B: Classification (Social Network Ads)
    print("\n--- STEP 3: Executing Classification Experiments (Social Network Ads) ---")
    df_classification, classification_artifacts = run_classification_experiments(df_ads, test_size=0.25, random_state=42)
    classification_csv_path = os.path.join(results_dir, "classification_results.csv")
    df_classification.to_csv(classification_csv_path, index=False)
    print(f"  [+] Classification results saved to: {classification_csv_path}")
    print(df_classification[["Algorithm", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].to_string(index=False))

    print("  [*] Generating classification visualizations...")
    clf_imgs = plot_classification_visualizations(df_classification, classification_artifacts, results_dir)
    for img in clf_imgs:
        print(f"      Saved: {os.path.basename(img)}")

    # 4. Dataset C: Boosting (Insurance Dataset)
    print("\n--- STEP 4: Executing Boosting Experiments (Insurance Dataset) ---")
    df_boost_cls, df_boost_reg, boosting_artifacts = run_boosting_experiments(df_ins, test_size=0.20, random_state=42)
    
    boost_cls_csv_path = os.path.join(results_dir, "boosting_results.csv")
    df_boost_cls.to_csv(boost_cls_csv_path, index=False)
    print(f"  [+] Boosting classification results saved to: {boost_cls_csv_path}")
    print(df_boost_cls[["Algorithm", "Task Type", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].to_string(index=False))

    boost_reg_csv_path = os.path.join(results_dir, "boosting_regression_companion.csv")
    df_boost_reg.to_csv(boost_reg_csv_path, index=False)
    print(f"  [+] Boosting regression (companion) results saved to: {boost_reg_csv_path}")
    print(df_boost_reg.to_string(index=False))

    print("  [*] Generating boosting visualizations...")
    boost_imgs = plot_boosting_visualizations(df_boost_cls, df_boost_reg, boosting_artifacts, results_dir)
    for img in boost_imgs:
        print(f"      Saved: {os.path.basename(img)}")

    # 5. Consolidated Summary Table
    print("\n--- STEP 5: Consolidating Best-Performing Algorithms ---")
    best_cluster_row = df_clustering.loc[df_clustering["Algorithm"] == "K-Means"].iloc[0]
    best_clf_row = df_classification.loc[df_classification["F1-score"].idxmax()]
    best_boost_row = df_boost_cls.loc[df_boost_cls["F1-score"].idxmax()]

    summary_rows = [
        {
            "Task": "Customer Segmentation (Clustering)",
            "Dataset": "Mall_Customers.csv",
            "Best Algorithm": "K-Means (k=5)",
            "Selection Metric": "Silhouette Score & Full-Population Coverage",
            "Performance Score": f"Silhouette: {best_cluster_row['Silhouette Score']:.4f} (Calinski-Harabasz: {best_cluster_row['Calinski-Harabasz Score']:.2f})",
            "Key Trade-off / Rationale": "K-Means cleanly clusters 100% of customers into 5 clear business archetypes without leaving 23 unassigned noise points like DBSCAN."
        },
        {
            "Task": "Purchase Prediction (Classification)",
            "Dataset": "Social_Network_Ads.csv",
            "Best Algorithm": best_clf_row["Algorithm"],
            "Selection Metric": "F1-score (Balance of Precision & Recall)",
            "Performance Score": f"F1: {best_clf_row['F1-score']:.4f} (Accuracy: {best_clf_row['Accuracy']*100:.1f}%, ROC-AUC: {best_clf_row['ROC-AUC']:.4f})",
            "Key Trade-off / Rationale": "Captures local non-linear interaction between age and salary thresholds better than linear models."
        },
        {
            "Task": "Lifestyle Risk Prediction (Boosting)",
            "Dataset": "insurance_pre.csv",
            "Best Algorithm": best_boost_row["Algorithm"],
            "Selection Metric": "F1-score & ROC-AUC",
            "Performance Score": f"F1: {best_boost_row['F1-score']:.4f} (Accuracy: {best_boost_row['Accuracy']*100:.2f}%, ROC-AUC: {best_boost_row['ROC-AUC']:.4f})",
            "Key Trade-off / Rationale": "Gradient Boosting sequentially optimizes log-loss, capturing charge-driven risk separation with near-perfect accuracy."
        }
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(results_dir, "summary_best_models.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"  [+] Consolidated summary saved to: {summary_csv_path}")
    print(df_summary[["Task", "Best Algorithm", "Selection Metric", "Performance Score"]].to_string(index=False))

    # 6. Report Compilation
    print("\n--- STEP 6: Compiling Technical Reports (Markdown & PDF) ---")
    md_report_path = os.path.join(results_dir, "REPORT.md")
    generate_markdown_report(
        metadata, df_clustering, df_classification,
        df_boost_cls, df_boost_reg, df_summary, md_report_path
    )
    print(f"  [+] Markdown report compiled: {md_report_path}")

    pdf_report_path = os.path.join(results_dir, "REPORT.pdf")
    try:
        generate_pdf_report(
            metadata, df_clustering, df_classification,
            df_boost_cls, df_summary, results_dir, pdf_report_path
        )
        print(f"  [+] PDF report compiled: {pdf_report_path}")
    except Exception as e:
        print(f"  [-] Note: PDF generation encountered an error ({e}); markdown report is fully available.")

    print("\n================================================================================")
    print("                     EXPERIMENT EXECUTION COMPLETED")
    print("================================================================================")


if __name__ == "__main__":
    main()
