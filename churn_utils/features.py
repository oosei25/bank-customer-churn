"""Feature engineering helpers for churn modeling."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

DEFAULT_TARGET_COL = "Exited"


# ---------- Core helpers ----------


def split_features_target(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a cleaned churn DataFrame into feature matrix X and target y.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned churn dataset (identifiers already removed).
    target_col : str, default "Exited"
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        All feature columns (no target).
    y : pd.Series
        Target labels as a 1D Series.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)  # ensure 0/1 ints
    return X, y


def basic_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical feature names from X.

    Numeric = int/float
    Categorical = object/category

    Returns
    -------
    numeric_features : list of str
    categorical_features : list of str
    """
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "Int64"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numeric_features, categorical_features


# ---------- Feature engineering ----------


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional, domain-inspired features for churn modeling.

    Assumes base columns roughly like:
    CreditScore, Age, Tenure, Balance, NumOfProducts,
    HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender

    Returns a new DataFrame with extra columns added.
    """
    df = df.copy()

    # 1. Credit score band (like bureau-style buckets)
    if "CreditScore" in df.columns:
        bins = [0, 580, 670, 740, 800, np.inf]
        labels = ["Poor", "Fair", "Good", "VeryGood", "Excellent"]
        df["CreditScoreBand"] = pd.cut(
            df["CreditScore"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        ).astype("category")

    # 2. Age group
    if "Age" in df.columns:
        age_bins = [0, 25, 35, 50, 65, np.inf]
        age_labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True,
        ).astype("category")

    # 3. Tenure band (relationship length)
    if "Tenure" in df.columns:
        tenure_bins = [-1, 1, 3, 5, 10, np.inf]
        tenure_labels = ["<1yr", "1-3yr", "3-5yr", "5-10yr", "10+yr"]
        df["TenureBand"] = pd.cut(
            df["Tenure"],
            bins=tenure_bins,
            labels=tenure_labels,
            include_lowest=True,
        ).astype("category")

    # 4. Balance to salary ratio (how "heavy" is the deposit)
    if "Balance" in df.columns and "EstimatedSalary" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["BalanceToSalary"] = df["Balance"] / df["EstimatedSalary"].replace(
                {0: np.nan}
            )
        df["BalanceToSalary"] = df["BalanceToSalary"].fillna(0.0)

    # 5. Simple engagement score
    #    Counts how many "engagement" traits the customer has.
    #    (Has credit card, is active, multi-product)
    has_cols = []
    if "HasCrCard" in df.columns:
        has_cols.append("HasCrCard")
    if "IsActiveMember" in df.columns:
        has_cols.append("IsActiveMember")
    elif "ActiveMember" in df.columns:
        has_cols.append("ActiveMember")

    if has_cols or "NumOfProducts" in df.columns:
        # Cast to numeric in case they arrived as booleans
        temp = df.copy()
        for c in has_cols:
            temp[c] = pd.to_numeric(temp[c], errors="coerce").fillna(0)

        multi_product_flag = (
            (temp["NumOfProducts"] > 1).astype(int)
            if "NumOfProducts" in temp.columns
            else 0
        )

        engagement_components = [temp[c] for c in has_cols]
        if isinstance(multi_product_flag, pd.Series):
            engagement_components.append(multi_product_flag)

        if engagement_components:
            df["EngagementScore"] = sum(engagement_components)
        else:
            df["EngagementScore"] = 0

    return df


def prepare_model_matrix(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    High-level helper: add engineered features and split X, y.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with engineered columns.
    y : pd.Series
        Target labels.
    """
    df_fe = add_engineered_features(df)
    X, y = split_features_target(df_fe, target_col=target_col)
    return X, y
