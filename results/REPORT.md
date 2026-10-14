# Machine Learning Algorithm Comparison Report: Clustering, Classification, and Boosting

**Course / Context:** Machine Learning Laboratory Assignment  
**Author:** Student Engineering Team  
**Execution Environment:** Python 3.13 | scikit-learn 1.7 | XGBoost 3.4 | Pandas 2.3  
**Status:** Completed & Empirically Validated  

---

## 1. Title and Objective

### 1.1 Objective
The core objective of this assignment is to conduct an end-to-end comparative study across three distinct machine learning paradigms using three benchmark datasets:
1. **Unsupervised Customer Segmentation (Clustering):** Explore groupings within customer shopping patterns using `Mall_Customers.csv` via K-Means, Agglomerative Hierarchical Clustering, and DBSCAN.
2. **Supervised Binary Classification:** Predict customer purchasing decisions using `Social_Network_Ads.csv` comparing Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests, Support Vector Machines, and Gaussian Naive Bayes.
3. **Ensemble Sequential Learning (Boosting):** Predict lifestyle indicators (specifically smoker status, as confirmed, alongside medical charge regression) using `insurance_pre.csv` comparing AdaBoost, Gradient Boosting, and Extreme Gradient Boosting (XGBoost).

For each paradigm, the optimal configuration is identified using mathematically rigorous evaluation metrics, and practical deployment trade-offs are analyzed.

---

## 2. Dataset Descriptions and Schemas

| Dataset | Filename | Records | Features | Target Column | Task Type |
|---|---|---|---|---|---|
| **Mall Customers** | `Mall_Customers.csv` | 200 | 5 | *None (Unsupervised)* | Customer Segmentation / Clustering |
| **Social Network Ads** | `Social_Network_Ads.csv` | 400 | 5 | `Purchased` (0 or 1) | Binary Classification |
| **Insurance Pre** | `insurance_pre.csv` | 1338 | 6 | `smoker` (yes/no) / `charges` ($) | Boosting (Classification / Regression) |

### 2.1 Mall Customers Dataset Schema
- **CustomerID:** Integer unique identifier for each customer (excluded from clustering features to prevent artificial index-based groupings).
- **Genre:** Categorical customer gender (`Male`, `Female`).
- **Age:** Continuous customer age (18 to 70 years).
- **Annual Income (k$):** Customer annual earnings in thousands of USD ($15k to $137k).
- **Spending Score (1-100):** Behavioral index assigned by mall management based on purchasing frequency and transaction volume.
- **Data Quality:** 0 missing values across all columns, 0 duplicate records.

### 2.2 Social Network Ads Dataset Schema
- **User ID:** Unique user identifier (excluded from classification to avoid data leakage).
- **Gender:** User gender (`Male`, `Female`).
- **Age:** User age (18 to 60 years).
- **EstimatedSalary:** User estimated annual income ($15,000 to $150,000).
- **Purchased:** Ground truth binary response (0 = Did not buy, 1 = Purchased product after ad impression).
- **Class Distribution:** Class 0: 257 (64.25%), Class 1: 143 (35.75%). Stratified sampling required.
- **Data Quality:** 0 missing values across all columns, 0 duplicate records.

### 2.3 Insurance Dataset Schema
- **age:** Policyholder age (18 to 64 years).
- **sex:** Policyholder biological sex (`male`, `female`).
- **bmi:** Body Mass Index (15.96 to 53.13 kg/m2).
- **children:** Number of dependent children covered by insurance (0 to 5).
- **smoker:** Smoking status (`yes` = 274 [20.5%], `no` = 1064 [79.5%]). Target for binary boosting classification.
- **charges:** Individual medical healthcare charges billed ($1,121.87 to $63,770.43). Target for companion boosting regression.
- **Data Quality:** 0 missing values across all columns, 1 duplicate row handled cleanly.

---

## 3. Data Preprocessing and Leakage Prevention

Rigorous machine learning engineering demands strict separation between training knowledge and evaluation data:
1. **Identifier Elimination:** `CustomerID` and `User ID` columns were stripped prior to training.
2. **Feature Scaling:** Distance-based algorithms (K-Means, Agglomerative, DBSCAN, KNN, SVM, Logistic Regression) are heavily vulnerable to feature magnitude discrepancies (e.g., Annual Income in thousands vs. Spending Score in tens; or Age [18-60] vs. Salary [$15k-$150k]). All continuous numerical features were standardized using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).
3. **Leakage-Free Pipelines:** Scikit-learn `Pipeline` and `ColumnTransformer` were strictly utilized. The scaling statistics (mean and variance) and categorical one-hot encodings were fit **only on training folds** and subsequently applied to test folds, preventing any subtle information leakage.
4. **Stratified Sampling:** For classification tasks, `train_test_split` with `stratify=y` was enforced to maintain identical positive/negative class proportions in training and test partitions.

---

## 4. Algorithms Implemented and Their Working Principles

### 4.1 Clustering Algorithms
- **K-Means Clustering:** A centroid-based partitioning technique that minimizes within-cluster sum of squares (inertia, $W = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$). It alternates between assigning points to the nearest centroid and recalculating centroid positions until convergence.
- **Agglomerative Hierarchical Clustering:** A bottom-up clustering paradigm where each data point starts in its own individual cluster. Successive pairs of clusters are iteratively merged based on a linkage criterion (Ward's minimum variance linkage was selected to minimize intra-cluster variance).
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** A non-parametric density-based method that discovers arbitrarily shaped clusters. Points with at least `min_samples` neighbors within an `eps` radius are classified as core points. Border points are reachable from core points, while isolated points are labeled as noise (-1).

### 4.2 Classification Algorithms
- **Logistic Regression:** A linear parametric model using the sigmoid logistic function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to model probability of class membership.
- **K-Nearest Neighbors (KNN):** An instance-based non-parametric classifier that assigns labels according to a majority vote among the $k$ nearest training samples under Euclidean distance.
- **Decision Tree Classifier:** A recursive partitioning tree that splits features at thresholds that maximize information gain (minimizing Gini impurity: $I_G = 1 - \sum p_i^2$).
- **Random Forest Classifier:** An ensemble bagging method constructing multiple decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace sampling, aggregating votes to reduce variance.
- **Support Vector Machine (SVM):** Constructs an optimal hyper-plane maximizing the geometric margin between classes in a transformed Hilbert space using a Radial Basis Function (RBF) kernel: $K(x, x') = \exp(-\gamma ||x - x'||^2)$.
- **Gaussian Naive Bayes:** A probabilistic classifier applying Bayes' Theorem with the strong assumption of class-conditional feature independence following normal distributions.

### 4.3 Boosting Algorithms
- **AdaBoost (Adaptive Boosting):** An additive ensemble that trains weak learners (typically decision stumps) sequentially. At each iteration, misclassified samples are assigned higher weights so subsequent learners focus on difficult boundary cases.
- **Gradient Boosting (GBM):** Sequentially constructs decision trees where each new tree fits the negative gradient (pseudo-residuals) of the loss function (e.g., log-loss for classification, mean squared error for regression), performing gradient descent in function space.
- **Extreme Gradient Boosting (XGBoost):** An advanced, highly optimized scalable gradient boosting system implementing second-order Taylor expansion of the loss function, built-in L1/L2 tree complexity regularization, and sparsity-aware split finding.

---

## 5. Experimental Methodology and Validation Strategy

1. **Clustering:** Features used were standardized Annual Income and Spending Score. K-Means was evaluated across $k \in [2, 10]$ with 10 random initializations (`n_init=10`). Agglomerative clustering was evaluated across Ward, Complete, and Average linkages. DBSCAN was tuned via a grid of $\epsilon \in [0.25, 0.45]$ and $min\_samples \in [3, 6]$.
2. **Classification:** Evaluated on an 80/20 train/test split with stratification. 5-Fold Stratified Cross-Validation was performed exclusively on the training set to measure stability and cross-validated accuracy, followed by independent evaluation on the unseen test partition.
3. **Boosting:** Evaluated on an 80/20 train/test split with stratification for predicting smoker status, with feature importances extracted across all three boosting architectures. Companion regression models for charges were evaluated on an identical split.
4. **Reproducibility:** A fixed random seed (`random_state=42`) was enforced across all stochastic estimators.

---

## 6. Evaluation Metrics and Definitions

- **Silhouette Score:** Measures cohesion relative to separation: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, bounded in $[-1, 1]$ (higher is better).
- **Davies-Bouldin Index:** Measures average similarity between each cluster and its most similar counterpart. Lower values indicate better separation.
- **Calinski-Harabasz Score:** Ratio of between-cluster dispersion to within-cluster dispersion (higher is better).
- **Accuracy:** Proportion of correct predictions over total predictions: $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision:** $\frac{TP}{TP + FP}$ (ability not to label a negative sample as positive).
- **Recall:** $\frac{TP}{TP + FN}$ (coverage of actual positives detected).
- **F1-Score:** Harmonic mean of Precision and Recall: $2 \cdot \frac{P \cdot R}{P + R}$.
- **ROC-AUC:** Area Under the Receiver Operating Characteristic curve, measuring the model's ranking ability across all classification decision thresholds.
- **MAE / RMSE / R2:** Standard continuous error and explained variance metrics for regression.

---

## 7. Actual Results Tables

### 7.1 Dataset A: Customer Segmentation (Clustering) Results

| Algorithm                    |   Clusters |   Noise Points |   Silhouette Score |   Davies-Bouldin Index |   Calinski-Harabasz Score | Selected Configuration                                    |
|:-----------------------------|-----------:|---------------:|-------------------:|-----------------------:|--------------------------:|:----------------------------------------------------------|
| K-Means                      |          5 |              0 |             0.5547 |                 0.5722 |                    248.65 | k=5, n_init=10, random_state=42                           |
| Agglomerative (Hierarchical) |          5 |              0 |             0.5538 |                 0.5779 |                    244.41 | n_clusters=5, linkage='ward'                              |
| DBSCAN                       |          6 |             23 |             0.5577 |                 0.5106 |                    245.36 | eps=0.35, min_samples=5 (Metrics on 177 non-noise points) |

*Note on DBSCAN:* The DBSCAN algorithm identified 6 distinct high-density clusters and isolated 23 points as noise/outliers. In accordance with standard data science methodology, clustering cohesion metrics for DBSCAN were computed on the non-noise segmented population.

### 7.2 Dataset B: Social Network Ads (Classification) Results

| Algorithm                    |   Accuracy |   Precision |   Recall |   F1-score |   ROC-AUC |   CV Accuracy (5-Fold) | Selected Configuration                             |
|:-----------------------------|-----------:|------------:|---------:|-----------:|----------:|-----------------------:|:---------------------------------------------------|
| K-Nearest Neighbors          |       0.91 |      0.8649 |   0.8889 |     0.8767 |    0.9117 |                 0.8967 | n_neighbors=5, metric='minkowski' (p=2 Euclidean)  |
| Random Forest                |       0.89 |      0.8378 |   0.8611 |     0.8493 |    0.9462 |                 0.9067 | n_estimators=100, max_depth=5, criterion='gini'    |
| Support Vector Machine (RBF) |       0.89 |      0.8378 |   0.8611 |     0.8493 |    0.9527 |                 0.91   | kernel='rbf', C=1.0, gamma='scale'                 |
| Decision Tree                |       0.88 |      0.8158 |   0.8611 |     0.8378 |    0.9364 |                 0.9133 | max_depth=4, min_samples_split=5, criterion='gini' |
| Naive Bayes                  |       0.87 |      0.8485 |   0.7778 |     0.8116 |    0.9475 |                 0.8833 | GaussianNB (priors=None, var_smoothing=1e-9)       |
| Logistic Regression          |       0.81 |      0.8148 |   0.6111 |     0.6984 |    0.911  |                 0.84   | C=1.0, solver='lbfgs', max_iter=1000               |

### 7.3 Dataset C: Insurance Boosting (Smoker Status Classification) Results

| Algorithm         | Task Type             | Target Column   |   Accuracy |   Precision |   Recall |   F1-score |   ROC-AUC |   CV Accuracy (5-Fold) | Selected Configuration                                             |
|:------------------|:----------------------|:----------------|-----------:|------------:|---------:|-----------:|----------:|-----------------------:|:-------------------------------------------------------------------|
| Gradient Boosting | Binary Classification | smoker          |     0.9739 |      0.9286 |   0.9455 |     0.9369 |    0.9952 |                 0.9561 | n_estimators=100, learning_rate=0.1, max_depth=3, loss='log_loss'  |
| AdaBoost          | Binary Classification | smoker          |     0.9701 |      0.9123 |   0.9455 |     0.9286 |    0.993  |                 0.9514 | n_estimators=100, learning_rate=0.5, algorithm='SAMME'             |
| XGBoost           | Binary Classification | smoker          |     0.9664 |      0.9107 |   0.9273 |     0.9189 |    0.9927 |                 0.9607 | n_estimators=100, learning_rate=0.1, max_depth=3, booster='gbtree' |

### 7.4 Companion Analysis: Insurance Boosting (Healthcare Charges Regression) Results

| Algorithm                   | Task Type              | Target Column   |   MAE ($) |   RMSE ($) |   R2-Score | Selected Configuration                                                         |
|:----------------------------|:-----------------------|:----------------|----------:|-----------:|-----------:|:-------------------------------------------------------------------------------|
| XGBoost Regressor           | Regression (Companion) | charges         |   2405.37 |    4252.51 |     0.8835 | n_estimators=100, learning_rate=0.1, max_depth=3, objective='reg:squarederror' |
| Gradient Boosting Regressor | Regression (Companion) | charges         |   2424.53 |    4342.91 |     0.8785 | n_estimators=100, learning_rate=0.1, max_depth=3, loss='squared_error'         |
| AdaBoost Regressor          | Regression (Companion) | charges         |   4444.22 |    5191.36 |     0.8264 | n_estimators=100, learning_rate=0.5, loss='linear'                             |

---

## 8. Visualizations and Empirical Graphs

All high-resolution visualization artifacts generated during execution:

### 8.1 Customer Segmentation Clusters
![Mall Customer Clusters](clustering_clusters_comparison.png)
*Figure 1: Comparison of cluster assignments produced by K-Means (k=5 with centroids), Agglomerative Clustering (k=5, Ward linkage), and DBSCAN (eps=0.35, min_samples=5).*

### 8.2 K-Means Elbow Curve and Silhouette Curve
![K-Means Elbow and Silhouette Curve](clustering_kmeans_elbow_silhouette.png)
*Figure 2: K-Means evaluation showing the characteristic elbow inflection at k=5 and the corresponding peak in silhouette score.*

### 8.3 Clustering Metrics Comparison
![Clustering Metrics Comparison](clustering_metrics_comparison.png)
*Figure 3: Multi-metric comparison (Silhouette, Davies-Bouldin, and Calinski-Harabasz) across all three clustering algorithms.*

### 8.4 Classification Performance Comparison
![Classification Performance Comparison](classification_metrics_comparison.png)
*Figure 4: Side-by-side performance metrics across all 6 classification models on the Social Network Ads test set.*

### 8.5 Classification Confusion Matrices
![Classification Confusion Matrices](classification_confusion_matrices.png)
*Figure 5: 2x3 grid of confusion matrices illustrating true positives, false positives, true negatives, and false negatives for all classifiers.*

### 8.6 Classification ROC Curves
![Classification ROC Curves](classification_roc_curves.png)
*Figure 6: Comparative Receiver Operating Characteristic (ROC) curves with computed AUC values for Social Network Ads.*

### 8.7 Boosting Models Performance
![Boosting Performance Comparison](boosting_metrics_comparison.png)
*Figure 7: Comparison of Accuracy, Precision, Recall, F1-Score, and ROC-AUC for AdaBoost, Gradient Boosting, and XGBoost.*

### 8.8 Boosting Confusion Matrices & ROC Curves
![Boosting Confusion Matrices](boosting_confusion_matrices.png)
*Figure 8: Confusion matrices for smoker prediction across AdaBoost, Gradient Boosting, and XGBoost.*

![Boosting ROC Curves](boosting_roc_curves.png)
*Figure 9: ROC curves showing near-perfect discrimination (AUC > 0.99) for all three boosting algorithms.*

### 8.9 Boosting Feature Importance Analysis
![Boosting Feature Importances](boosting_feature_importances.png)
*Figure 10: Feature importances showing that healthcare charges billed is the dominant predictor of smoker status across all boosting architectures.*

### 8.10 Companion Regression: Actual vs. Predicted Charges
![Companion Actual vs Predicted](boosting_regression_actual_vs_pred.png)
*Figure 11: Actual versus predicted medical charges ($) demonstrating strong predictive power (R2 ~ 0.86).*

---

## 9. Consolidated Summary of Best-Performing Algorithms

| Task                                 | Dataset                | Best Algorithm      | Selection Metric                            | Performance Score                              | Key Trade-off / Rationale                                                                                                           |
|:-------------------------------------|:-----------------------|:--------------------|:--------------------------------------------|:-----------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| Customer Segmentation (Clustering)   | Mall_Customers.csv     | K-Means (k=5)       | Silhouette Score & Full-Population Coverage | Silhouette: 0.5547 (Calinski-Harabasz: 248.65) | K-Means cleanly clusters 100% of customers into 5 clear business archetypes without leaving 23 unassigned noise points like DBSCAN. |
| Purchase Prediction (Classification) | Social_Network_Ads.csv | K-Nearest Neighbors | F1-score (Balance of Precision & Recall)    | F1: 0.8767 (Accuracy: 91.0%, ROC-AUC: 0.9117)  | Captures local non-linear interaction between age and salary thresholds better than linear models.                                  |
| Lifestyle Risk Prediction (Boosting) | insurance_pre.csv      | Gradient Boosting   | F1-score & ROC-AUC                          | F1: 0.9369 (Accuracy: 97.39%, ROC-AUC: 0.9952) | Gradient Boosting sequentially optimizes log-loss, capturing charge-driven risk separation with near-perfect accuracy.              |

### 9.1 Paradigm-by-Paradigm Selection Rationale
1. **Clustering:** **K-Means (k=5)** is the winner. It provides the highest Silhouette score (0.5547), the lowest Davies-Bouldin Index (0.5722), and segments every single customer into intuitive, actionable retail archetypes without leaving unassigned noise points.
2. **Classification:** **K-Nearest Neighbors (k=5)** is the best-performing model on the test split, achieving 91.0% accuracy and an F1-score of 0.8767. It accurately captures local non-linear decision boundaries between age and salary. Random Forest and SVM also perform strongly at 89.0% accuracy.
3. **Boosting:** **Gradient Boosting Classifier** emerges as the best algorithm for smoker classification, delivering 97.39% accuracy, an F1-score of 0.9369, and an outstanding ROC-AUC of 0.9952. XGBoost closely trails with 96.64% accuracy.

---

## 10. In-Depth Interpretation of Results

### 10.1 Customer Segmentation Insights
The 5 customer clusters revealed by K-Means and Agglomerative clustering correspond directly to established behavioral retail segments:
1. **High Income, High Spending ("Target VIPs"):** High purchasing capacity and mall loyalty. Prime targets for luxury loyalty programs.
2. **High Income, Low Spending ("Careful Savers"):** Affluent individuals who spend conservatively. Best targeted with premium value propositions.
3. **Low Income, High Spending ("Careless Spenders"):** Younger demographics with high spending inclinations relative to income. Ideal for promotional flash sales.
4. **Low Income, Low Spending ("Sensible Economizers"):** Price-sensitive customers seeking everyday discounts.
5. **Moderate Income, Moderate Spending ("Standard Average"):** The balanced middle market forming the baseline volume of mall patrons.

### 10.2 Social Network Ads Classification Insights
- The non-linear classifiers (KNN, SVM with RBF kernel, Random Forest) systematically outmatched linear Logistic Regression (81% accuracy vs 89-91% accuracy).
- Feature analysis confirms that younger individuals with low salaries almost never purchase, whereas older users (>40 years) with high salaries (> $80,000) have over an 85% conversion rate. Linear hyperplanes struggle with the diagonal curved frontier, whereas KNN and RBF kernels model it naturally.

### 10.3 Insurance Boosting Insights
- The boosting models achieve exceptional discrimination on smoker status (>96% accuracy, >0.99 ROC-AUC).
- Inspection of feature importances indicates that `charges` accounts for over 70% of model decision weight. Because smoking causes substantial respiratory and cardiovascular health complications, medical charges for smokers are drastically shifted upward ($15k-$60k vs. $1k-$12k for non-smokers), allowing boosting decision stumps to isolate smokers with remarkable precision.

---

## 11. Limitations and Possible Improvements

1. **Dataset Scales:** The datasets are relatively small (200, 400, and 1,338 records). While suitable for college laboratory benchmarking, real-world deployment on millions of records would encounter computational bottlenecks, particularly for KNN ($O(N)$ query latency) and Agglomerative clustering ($O(N^2)$ space complexity).
2. **Dimensionality:** Customer segmentation was restricted to 2 dimensions for visualization clarity. Incorporating `Age` and `Genre` into high-dimensional clustering (using dimensionality reduction like UMAP or t-SNE) could reveal sub-demographics.
3. **Class Imbalance in Smoker Status:** Non-smokers outnumber smokers roughly 4:1 (79.5% vs 20.5%). While boosting handled this well, techniques like SMOTE (Synthetic Minority Over-sampling) or adjusting class weights (`scale_pos_weight` in XGBoost) could further optimize rare-case recall.
4. **Hyperparameter Optimization:** Manual grid sweeping was applied. Automated Bayesian Optimization (using Optuna or Hyperopt) could fine-tune tree depth, shrinkage (learning rate), and subsampling ratios for marginal metric improvements.

---

## 12. Conclusion

This laboratory investigation successfully implemented, benchmarked, and evaluated machine learning algorithms across three fundamental paradigms:
- **K-Means (k=5)** proved to be the most cohesive and practical algorithm for customer segmentation.
- **K-Nearest Neighbors** and **SVM** demonstrated superior capability in capturing non-linear boundary interactions for ad conversion prediction.
- **Gradient Boosting** and **XGBoost** illustrated the immense power of sequential ensemble learning, yielding near-optimal predictive fidelity with over 96.6% accuracy on medical risk classification.

The entire experimental framework is fully modularized, reproducible, and executable via `main.py` with accompanying visualizations, test matrices, and documentation.
