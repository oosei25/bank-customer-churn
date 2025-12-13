"""Model training and inference utilities."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from .features import basic_feature_lists, prepare_model_matrix


# ---------- Train / test split ----------


def train_test_split_stratified(
    X: pd.DataFrame,
    y,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience wrapper for a stratified train/test split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target labels.
    test_size : float, default 0.3
        Proportion of data to put in the test set.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# ---------- Pipelines ----------


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - imputes numeric features (median)
    - imputes categorical features (most frequent)
    - scales numeric features
    - one-hot encodes categoricals
    """
    numeric_features, categorical_features = basic_feature_lists(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_logreg_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Create a preprocessing + Logistic Regression pipeline.

    Uses:
    - StandardScaler for numeric features
    - OneHotEncoder for categoricals
    - L2-regularized LogisticRegression with class_weight='balanced'
    """
    preprocessor = _build_preprocessor(X)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def build_rf_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Create a preprocessing + RandomForestClassifier pipeline.

    Uses:
    - StandardScaler for numeric features (not strictly required but harmless)
    - OneHotEncoder for categoricals
    - RandomForest with conservative regularization to reduce overfitting
    """
    preprocessor = _build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def build_xgb_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Create a preprocessing + XGBoost pipeline.

    Uses:
    - StandardScaler for numeric features (not strictly required for trees)
    - OneHotEncoder for categoricals
    - XGBClassifier tuned conservatively to reduce overfitting
    """
    preprocessor = _build_preprocessor(X)

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=1.0,  # adjust if class imbalance is strong
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# ---------- High-level helper ----------


def prepare_data_and_pipelines(
    df: pd.DataFrame,
    target_col: str = "Exited",
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    High-level helper that:
    - Adds engineered features
    - Splits into train/test
    - Builds logreg, RF, and XGB pipelines

    This is a convenient entrypoint for notebooks.

    Returns
    -------
    X_train, X_test, y_train, y_test, logreg_pipe, rf_pipe, xgb_pipe
    """
    # Add engineered features & split X, y
    X, y = prepare_model_matrix(df, target_col=target_col)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build pipelines based on training columns
    logreg_pipe = build_logreg_pipeline(X_train)
    rf_pipe = build_rf_pipeline(X_train)
    xgb_pipe = build_xgb_pipeline(X_train)

    return X_train, X_test, y_train, y_test, logreg_pipe, rf_pipe, xgb_pipe
