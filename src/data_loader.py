"""
data_loader.py - Data ingestion, schema validation, and descriptive analysis.
"""

import os
from typing import Dict, Tuple, Any
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV file with validation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")
    return pd.read_csv(file_path)


def analyze_dataset(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Generate comprehensive schema analysis and statistics for a dataset."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    
    missing_dict = df.isnull().sum().to_dict()
    total_missing = sum(missing_dict.values())
    dup_count = int(df.duplicated().sum())

    stats = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "missing_counts": missing_dict,
        "total_missing": total_missing,
        "duplicate_rows": dup_count,
        "summary": df.describe(include="all").to_dict()
    }
    return stats


def load_all_datasets(data_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load Mall_Customers, Social_Network_Ads, and insurance_pre datasets.
    Returns dataframes and inspection metadata dictionary.
    """
    mall_path = os.path.join(data_dir, "Mall_Customers.csv")
    ads_path = os.path.join(data_dir, "Social_Network_Ads.csv")
    ins_path = os.path.join(data_dir, "insurance_pre.csv")

    df_mall = load_dataset(mall_path)
    df_ads = load_dataset(ads_path)
    df_ins = load_dataset(ins_path)

    metadata = {
        "mall": analyze_dataset(df_mall, "Mall Customers"),
        "ads": analyze_dataset(df_ads, "Social Network Ads"),
        "insurance": analyze_dataset(df_ins, "Insurance Charges & Lifestyle")
    }

    return df_mall, df_ads, df_ins, metadata
