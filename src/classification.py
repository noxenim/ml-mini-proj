"""
classification.py - Supervised classification on Social_Network_Ads dataset.
Implements Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and Naive Bayes
using scikit-learn Pipelines, ColumnTransformer, and Stratified Cross-Validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


def run_classification_experiments(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train and evaluate classification models on Social_Network_Ads.
    Prevents leakage by fitting ColumnTransformer exclusively inside Pipelines.
    """
    # Exclude User ID (identifier) to prevent leakage
    feature_cols = ["Gender", "Age", "EstimatedSalary"]
    target_col = "Purchased"

    X = df[feature_cols]
    y = df[target_col]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessing Pipeline (StandardScaler for numericals, OneHotEncoder for Gender)
    num_features = ["Age", "EstimatedSalary"]
    cat_features = ["Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
        ]
    )

    # Model Dictionary with reproducible random seeds
    models = {
        "Logistic Regression": (
            LogisticRegression(C=1.0, random_state=random_state),
            "C=1.0, solver='lbfgs', max_iter=1000"
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
            "n_neighbors=5, metric='minkowski' (p=2 Euclidean)"
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=random_state),
            "max_depth=4, min_samples_split=5, criterion='gini'"
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state),
            "n_estimators=100, max_depth=5, criterion='gini'"
        ),
        "Support Vector Machine (RBF)": (
            SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state),
            "kernel='rbf', C=1.0, gamma='scale'"
        ),
        "Naive Bayes": (
            GaussianNB(),
            "GaussianNB (priors=None, var_smoothing=1e-9)"
        )
    }

    results = []
    trained_artifacts = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": {},
        "predictions": {},
        "probabilities": {},
        "confusion_matrices": {},
        "roc_curves": {},
        "cv_scores": {}
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, (clf, config_str) in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # 5-Fold Stratified Cross-Validation on Training Set
        cv_res = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1", "roc_auc"]
        )
        mean_cv_acc = float(np.mean(cv_res["test_accuracy"]))
        mean_cv_f1 = float(np.mean(cv_res["test_f1"]))
        mean_cv_roc = float(np.mean(cv_res["test_roc_auc"]))

        # Fit on entire Training Set
        pipeline.fit(X_train, y_train)

        # Evaluate on Hold-out Test Set
        y_pred = pipeline.predict(X_test)
        
        # Predicted probabilities for ROC-AUC
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            y_prob = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Record artifacts
        trained_artifacts["models"][name] = pipeline
        trained_artifacts["predictions"][name] = y_pred
        trained_artifacts["probabilities"][name] = y_prob
        trained_artifacts["confusion_matrices"][name] = cm
        trained_artifacts["roc_curves"][name] = (fpr, tpr, roc_auc)
        trained_artifacts["cv_scores"][name] = {
            "cv_accuracy": mean_cv_acc,
            "cv_f1": mean_cv_f1,
            "cv_roc_auc": mean_cv_roc
        }

        results.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "CV Accuracy (5-Fold)": round(mean_cv_acc, 4),
            "Selected Configuration": config_str
        })

    df_results = pd.DataFrame(results).sort_values(by="F1-score", ascending=False).reset_index(drop=True)
    return df_results, trained_artifacts
