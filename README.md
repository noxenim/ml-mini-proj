# Machine Learning Algorithm Comparison Project

An end-to-end laboratory assignment benchmarking machine learning algorithms across three fundamental paradigms:
1. **Unsupervised Customer Segmentation (Clustering):** `Mall_Customers.csv`
2. **Supervised Binary Classification:** `Social_Network_Ads.csv`
3. **Sequential Ensemble Learning (Boosting):** `insurance_pre.csv`

---

## 1. Project Directory Structure

```
mlproj/
├── Mall_Customers.csv               # Dataset A: 200 customer records (Income, Spending Score)
├── Social_Network_Ads.csv           # Dataset B: 400 ad audience records (Age, Salary, Purchased)
├── insurance_pre.csv                # Dataset C: 1,338 insurance policies (Demographics, Smoker, Charges)
├── requirements.txt                 # Exact dependencies for ML, plotting, and report generation
├── main.py                          # Unified executable entry point running all experiments
├── generate_notebook.py             # Script to build experiments.ipynb
├── experiments.ipynb                # Fully executed Jupyter notebook with outputs & plots
├── PRESENTATION_GUIDE.md            # Defense guide, script, and Viva Voce Q&A for the teacher
├── README.md                        # Setup and reproduction guide (this file)
├── src/                             # Modular Python package
│   ├── __init__.py
│   ├── data_loader.py               # Data ingestion, schema validation, and descriptive stats
│   ├── clustering.py                # K-Means, Agglomerative, and DBSCAN implementations
│   ├── classification.py            # Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes
│   ├── boosting.py                  # AdaBoost, Gradient Boosting, XGBoost (Classification & Regression)
│   ├── visualization.py             # 300 DPI publication-grade Seaborn/Matplotlib routines
│   └── report_generator.py          # Automated Markdown and PDF report compilation
└── results/                         # Generated tables, visual artifacts, and reports
    ├── clustering_results.csv       # Clustering metrics comparison table
    ├── classification_results.csv   # Classification metrics comparison table
    ├── boosting_results.csv         # Boosting smoker classification table
    ├── boosting_regression_companion.csv # Boosting charges regression table
    ├── summary_best_models.csv      # Consolidated summary of best models
    ├── REPORT.md                    # Comprehensive 12-section technical report in Markdown
    ├── REPORT.pdf                   # Compiled 5-page PDF report with embedded figures
    └── *.png                        # 10 High-resolution visualization charts
```

---

## 2. Environment Setup & Installation

### Option A: Using Existing Python Environment (Python 3.10+)
Install the required dependencies directly:
```bash
pip install -r requirements.txt
```

### Option B: Using a Clean Virtual Environment (Recommended)
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. How to Run the Project

### Execute the Complete Pipeline (One Command)
Run the unified entry point:
```bash
python main.py
```
This single command will:
1. Ingest and validate all three datasets.
2. Train and evaluate all clustering, classification, and boosting algorithms.
3. Export all CSV tables into `results/`.
4. Render and save all 10 high-resolution charts (300 DPI) into `results/`.
5. Compile `results/REPORT.md` and generate `results/REPORT.pdf`.
6. Display formatted console summaries.

### Interactive Exploration via Jupyter Notebook
Launch the notebook to inspect interactive code cells, outputs, and visualizations:
```bash
jupyter notebook experiments.ipynb
```
*(Or open `experiments.ipynb` directly in VS Code / IDE with the Jupyter extension).*

---

## 4. Key Results & Summary of Best-Performing Models

| Task | Dataset | Best-Performing Algorithm | Selection Metric | Performance Score | Key Trade-off / Rationale |
|---|---|---|---|---|---|
| **Customer Segmentation (Clustering)** | `Mall_Customers.csv` | **K-Means ($k=5$)** | Silhouette Score & Full Coverage | **Silhouette: 0.5547** (Calinski-Harabasz: 248.65) | Partitions 100% of customers into 5 business archetypes without leaving 23 unassigned noise points like DBSCAN. |
| **Purchase Prediction (Classification)** | `Social_Network_Ads.csv` | **K-Nearest Neighbors ($k=5$)** | F1-Score | **F1: 0.8767** (Accuracy: 91.0%, ROC-AUC: 0.9117) | Naturally resolves the curved, non-linear age/salary conversion frontier where linear models fail. |
| **Lifestyle Risk Prediction (Boosting)** | `insurance_pre.csv` | **Gradient Boosting** | F1-Score & ROC-AUC | **F1: 0.9369** (Accuracy: 97.39%, ROC-AUC: 0.9952) | Sequentially minimizes log-loss pseudo-residuals, isolating smokers with remarkable accuracy via healthcare charges. |

---

## 5. Visualizations Generated

All figures are automatically saved in `results/` at 300 DPI:
1. `clustering_clusters_comparison.png`: 1x3 panel comparing K-Means centroids, Agglomerative clusters, and DBSCAN noise.
2. `clustering_kmeans_elbow_silhouette.png`: Dual-axis Inertia Elbow Curve and Silhouette score progression across $k \in [2, 10]$.
3. `clustering_metrics_comparison.png`: Grouped comparison of Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
4. `classification_metrics_comparison.png`: Accuracy, Precision, Recall, F1, and ROC-AUC comparison for 6 classifiers.
5. `classification_confusion_matrices.png`: 2x3 confusion matrix grid with counts and error breakdown.
6. `classification_roc_curves.png`: Multi-model ROC curves with AUC annotations against the chance line.
7. `boosting_metrics_comparison.png`: Side-by-side metric comparison across AdaBoost, Gradient Boosting, and XGBoost.
8. `boosting_confusion_matrices.png`: 1x3 confusion matrix comparison for smoker detection.
9. `boosting_roc_curves.png`: Near-ideal ROC curves illustrating $\text{AUC} > 0.99$.
10. `boosting_feature_importances.png`: Relative feature contribution showing charges as the dominant predictor.
11. `boosting_regression_actual_vs_pred.png`: Companion regression scatter plots of predicted vs. actual charges.

---

## 6. How to Present to Your Teacher

Refer to the dedicated guide: **[`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md)**  
It contains:
- An itemized explanation of every decision made from scratch and its rationale.
- A word-for-word spoken presentation script (with slide-by-slide timing).
- Common Viva Voce / defense questions and model answers.
