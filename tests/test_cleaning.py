import pandas as pd

from churn_utils.cleaning import clean_churn_dataframe


def test_clean_churn_dataframe_basic():
    # minimal messy example similar to Bank_Churn_Messy.xlsx join
    df_raw = pd.DataFrame(
        {
            "CustomerId": [1, 2],
            "Geography": ["FRA", "Germany"],
            "Gender": ["Female", "Male"],
            "Age": [42, 50],
            "Tenure": [5, 10],
            "Balance": ["1,000.00", "0.0"],   # messy strings
            "NumOfProducts": [1, 2],
            "HasCrCard": ["Yes", "No"],
            "IsActiveMember": ["Yes", "No"],
            "EstimatedSalary": ["50,000.00", "60,000.00"],
            "Exited": [0, 1],
        }
    )

    df_clean = clean_churn_dataframe(df_raw)

    # Required columns exist
    expected_cols = {
        "Age",
        "Geography",
        "Gender",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    }
    assert expected_cols.issubset(df_clean.columns)

    # Types look numeric where expected
    numeric_cols = [
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    assert all(pd.api.types.is_numeric_dtype(df_clean[c]) for c in numeric_cols)

    # Geography codes normalized (e.g., "FRA" -> "France")
    assert "France" in df_clean["Geography"].unique()