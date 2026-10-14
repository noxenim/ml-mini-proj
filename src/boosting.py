"""
boosting.py - Boosting algorithms evaluation on insurance_pre dataset.
Implements AdaBoost, Gradient Boosting, and XGBoost for binary classification
(predicting smoker status as confirmed by user), plus companion regression analysis (predicting charges).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb


def run_boosting_experiments(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Train and evaluate AdaBoost, Gradient Boosting, and XGBoost.
    Primary Task: Binary Classification predicting 'smoker' status.
    Companion Task: Continuous Regression predicting healthcare 'charges'.
    """
    # -------------------------------------------------------------
    # PRIMARY TASK: CLASSIFICATION (Predicting Smoker Status)
    # -------------------------------------------------------------
    num_features_cls = ["age", "bmi", "children", "charges"]
    cat_features_cls = ["sex"]
    target_cls = "smoker"

    X_cls = df[num_features_cls + cat_features_cls]
    y_cls = (df[target_cls].str.strip().str.lower() == "yes").astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=test_size, random_state=random_state, stratify=y_cls
    )

    preprocessor_cls = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features_cls),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features_cls)
        ]
    )

    boosting_classifiers = {
        "AdaBoost": (
            AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=random_state),
            "n_estimators=100, learning_rate=0.5, algorithm='SAMME'"
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
            "n_estimators=100, learning_rate=0.1, max_depth=3, loss='log_loss'"
        ),
        "XGBoost": (
            xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric="logloss", random_state=random_state),
            "n_estimators=100, learning_rate=0.1, max_depth=3, booster='gbtree'"
        )
    }

    cls_results = []
    cls_artifacts = {
        "X_train": X_train_c,
        "X_test": X_test_c,
        "y_train": y_train_c,
        "y_test": y_test_c,
        "models": {},
        "predictions": {},
        "probabilities": {},
        "confusion_matrices": {},
        "roc_curves": {},
        "feature_importances": {}
    }

    cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, (clf, config_str) in boosting_classifiers.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor_cls),
            ("classifier", clf)
        ])

        # 5-Fold Stratified CV on training set
        cv_res = cross_validate(pipe, X_train_c, y_train_c, cv=cv_cls, scoring=["accuracy", "f1", "roc_auc"])
        mean_cv_acc = float(np.mean(cv_res["test_accuracy"]))
        mean_cv_f1 = float(np.mean(cv_res["test_f1"]))

        # Fit on training set
        pipe.fit(X_train_c, y_train_c)

        # Predict on test set
        y_pred = pipe.predict(X_test_c)
        y_prob = pipe.predict_proba(X_test_c)[:, 1]

        acc = float(accuracy_score(y_test_c, y_pred))
        prec = float(precision_score(y_test_c, y_pred, zero_division=0))
        rec = float(recall_score(y_test_c, y_pred, zero_division=0))
        f1 = float(f1_score(y_test_c, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test_c, y_prob))
        cm = confusion_matrix(y_test_c, y_pred)
        fpr, tpr, _ = roc_curve(y_test_c, y_prob)

        # Feature importances extraction
        model_step = pipe.named_steps["classifier"]
        feature_names = num_features_cls + ["sex_male"]
        importances = model_step.feature_importances_

        cls_artifacts["models"][name] = pipe
        cls_artifacts["predictions"][name] = y_pred
        cls_artifacts["probabilities"][name] = y_prob
        cls_artifacts["confusion_matrices"][name] = cm
        cls_artifacts["roc_curves"][name] = (fpr, tpr, roc_auc)
        cls_artifacts["feature_importances"][name] = dict(zip(feature_names, importances))

        cls_results.append({
            "Algorithm": name,
            "Task Type": "Binary Classification",
            "Target Column": "smoker",
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "CV Accuracy (5-Fold)": round(mean_cv_acc, 4),
            "Selected Configuration": config_str
        })

    df_cls_results = pd.DataFrame(cls_results).sort_values(by="F1-score", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------
    # COMPANION TASK: REGRESSION (Predicting Healthcare Charges)
    # -------------------------------------------------------------
    num_features_reg = ["age", "bmi", "children"]
    cat_features_reg = ["sex", "smoker"]
    target_reg = "charges"

    X_reg = df[num_features_reg + cat_features_reg]
    y_reg = df[target_reg].values

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=test_size, random_state=random_state
    )

    preprocessor_reg = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features_reg),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features_reg)
        ]
    )

    boosting_regressors = {
        "AdaBoost Regressor": (
            AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=random_state),
            "n_estimators=100, learning_rate=0.5, loss='linear'"
        ),
        "Gradient Boosting Regressor": (
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
            "n_estimators=100, learning_rate=0.1, max_depth=3, loss='squared_error'"
        ),
        "XGBoost Regressor": (
            xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
            "n_estimators=100, learning_rate=0.1, max_depth=3, objective='reg:squarederror'"
        )
    }

    reg_results = []
    reg_artifacts = {
        "X_train": X_train_r,
        "X_test": X_test_r,
        "y_train": y_train_r,
        "y_test": y_test_r,
        "models": {},
        "predictions": {}
    }

    for name, (reg, config_str) in boosting_regressors.items():
        pipe_r = Pipeline([
            ("preprocessor", preprocessor_reg),
            ("regressor", reg)
        ])
        pipe_r.fit(X_train_r, y_train_r)
        y_pred_r = pipe_r.predict(X_test_r)

        mae = float(mean_absolute_error(y_test_r, y_pred_r))
        rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
        r2 = float(r2_score(y_test_r, y_pred_r))

        reg_artifacts["models"][name] = pipe_r
        reg_artifacts["predictions"][name] = y_pred_r

        reg_results.append({
            "Algorithm": name,
            "Task Type": "Regression (Companion)",
            "Target Column": "charges",
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "R2-Score": round(r2, 4),
            "Selected Configuration": config_str
        })

    df_reg_results = pd.DataFrame(reg_results).sort_values(by="R2-Score", ascending=False).reset_index(drop=True)

    combined_artifacts = {
        "classification": cls_artifacts,
        "regression": reg_artifacts
    }

    return df_cls_results, df_reg_results, combined_artifacts
