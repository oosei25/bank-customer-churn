"""Visualization helpers for churn modeling."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


def plot_churn_rate(df: pd.DataFrame) -> None:
    """Print overall churn rate."""
    churn_rate = df["Exited"].mean()
    print(f"Overall churn rate: {churn_rate:.2%}")


def barplot_churn_by(
    df: pd.DataFrame,
    col: str,
    col_label: str | None = None,
    value_labels: dict | None = None,
    palette: str = "viridis",
) -> None:
    """
    Bar plot of churn rate by a categorical column.
    """
    col_label = col_label or col

    grouped = df.groupby(col)["Exited"].mean().reset_index()

    # If we want nicer tick labels for the categories
    if value_labels is not None:
        grouped["__display__"] = grouped[col].map(value_labels)
        x_col = "__display__"
    else:
        x_col = col

    plt.figure(figsize=(6, 4))
    sns.barplot(data=grouped, x=x_col, y="Exited", palette=palette)
    plt.ylabel("Churn rate")
    plt.xlabel(col_label)
    plt.title(f"Churn rate by {col_label}")
    plt.tight_layout()
    plt.show()


def distplot_feature(
    df: pd.DataFrame,
    col: str,
    col_label: str | None = None,
    palette: str = "viridis",
) -> None:
    """
    Distribution of a numeric feature split by churn.
    """
    col_label = col_label or col

    plt.figure(figsize=(6, 4))
    sns.kdeplot(
        data=df,
        x=col,
        hue="Exited",
        common_norm=False,
        fill=True,
        palette=palette,
    )
    plt.xlabel(col_label)
    plt.title(f"Distribution of {col_label} by churn")
    plt.tight_layout()
    plt.show()
