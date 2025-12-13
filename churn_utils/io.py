"""Data IO helpers for the churn project."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def save_processed_data(
    df: pd.DataFrame, filename: str = "bank_churn_clean1.csv"
) -> None:
    """Save cleaned/processed data to data/processed."""
    path = DATA_DIR / "processed" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and remove leading ':' from column names.
    Helps fix headers like ':customerId'.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(":", "", regex=False)
    return df


def load_messy_churn_excel(
    filename: str = "Bank_Churn_Messy.xlsx",
    customer_sheet: str = "Customer_Info",
    account_sheet: str = "Account_Info",
) -> pd.DataFrame:
    """
    Load the messy Excel workbook and merge the two sheets into a single DataFrame.

    - Reads `Customer_Info` (CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, EstimatedSalary)
    - Reads `Account_Info` (CustomerId, Balance, NumOfProducts, HasCrCard, Tenure, IsActiveMember, Exited)
    - Standardizes column names, fixes ':customerId'
    - Merges on CustomerId
    - Resolves duplicate Tenure columns into a single 'Tenure'
    """
    path = DATA_DIR / "raw" / filename
    xls = pd.ExcelFile(path)

    # Read sheets
    df_cust = pd.read_excel(xls, sheet_name=customer_sheet)
    df_acct = pd.read_excel(xls, sheet_name=account_sheet)

    # Standardize column names (strip, remove ':')
    df_cust = _standardize_columns(df_cust)
    df_acct = _standardize_columns(df_acct)

    # Ensure the key is exactly 'CustomerId'
    if "CustomerId" not in df_cust.columns:
        raise KeyError("CustomerId column not found in Customer_Info sheet")
    if "CustomerId" not in df_acct.columns:
        raise KeyError("CustomerId column not found in Account_Info sheet")

    # Merge on CustomerId
    df = pd.merge(
        df_cust, df_acct, on="CustomerId", how="inner", suffixes=("_cust", "_acct")
    )

    # Resolve Tenure columns if both exist
    if "Tenure_cust" in df.columns and "Tenure_acct" in df.columns:
        df["Tenure"] = df["Tenure_acct"].fillna(df["Tenure_cust"])
        df = df.drop(columns=["Tenure_cust", "Tenure_acct"])
    elif "Tenure_cust" in df.columns:
        df = df.rename(columns={"Tenure_cust": "Tenure"})
    elif "Tenure_acct" in df.columns:
        df = df.rename(columns={"Tenure_acct": "Tenure"})

    return df
