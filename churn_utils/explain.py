from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def make_shap_explainer(
    tree_pipeline,
    X_background: pd.DataFrame,
    max_background: int = 1000,
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    """
    Create a SHAP TreeExplainer for a fitted tree-based pipeline.

    Parameters
    ----------
    tree_pipeline
        Fitted sklearn Pipeline with "preprocess" and "model" (RF/XGB).
    X_background : pd.DataFrame
        Training data used as background (original feature space).
    max_background : int, default 1000
        Maximum number of background samples to use for SHAP
        to keep computation manageable.

    Returns
    -------
    explainer : shap.TreeExplainer
        SHAP explainer object.
    X_bg_transformed : np.ndarray
        Transformed background data in the model's feature space.
    feature_names : list[str] or None
    """
    # Sample background to keep SHAP fast
    if len(X_background) > max_background:
        X_bg = X_background.sample(n=max_background, random_state=42)
    else:
        X_bg = X_background

    # Transform background through the pipeline's preprocessor
    preprocessor = tree_pipeline.named_steps["preprocess"]
    X_bg_transformed = preprocessor.transform(X_bg)

    # Try to get feature names from the ColumnTransformer
    try:
        raw_names = preprocessor.get_feature_names_out()
        feature_names = [name.split("__", 1)[-1] for name in raw_names]
    except AttributeError:
        feature_names = None

    model = tree_pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)

    return explainer, X_bg_transformed, feature_names


def shap_summary_plot(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: Optional[list[str]] = None,
    max_display: int = 15,
) -> None:
    """
    Global SHAP summary plot for feature importance.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Fitted SHAP explainer.
    X_transformed : np.ndarray
        Transformed feature matrix used for SHAP values.
    feature_names : list of str, optional
        Names of transformed features (e.g., after one-hot encoding).
        If None, SHAP will try to infer them.
    max_display : int, default 15
        Maximum number of features to display.
    """
    shap_values = explainer.shap_values(X_transformed)
    # plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.show()
    plt.close()


def shap_local_explanation(
    explainer: shap.TreeExplainer,
    tree_pipeline,
    X_row: pd.DataFrame,
    feature_names: Optional[list[str]] = None,
):
    """
    Local SHAP explanation for a single instance.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Fitted SHAP explainer.
    tree_pipeline
        Fitted sklearn Pipeline.
    X_row : pd.DataFrame
        Single-row DataFrame of original features.
    feature_names : list of str, optional
        Names of transformed features; optional.
    """
    preprocessor = tree_pipeline.named_steps["preprocess"]
    X_row_transformed = preprocessor.transform(X_row)

    shap_values = explainer.shap_values(X_row_transformed)

    # Take the first (or only) class for binary classification
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    else:
        sv = shap_values

    shap.waterfall_plot = getattr(shap, "waterfall_plot", None)
    if shap.waterfall_plot is not None:
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=(
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                ),
                data=X_row_transformed[0],
                feature_names=feature_names,
            )
        )
    else:
        # Fallback to force_plot if waterfall is unavailable
        shap.force_plot(
            explainer.expected_value,
            sv[0],
            X_row_transformed[0],
            feature_names=feature_names,
            matplotlib=True,
        )
