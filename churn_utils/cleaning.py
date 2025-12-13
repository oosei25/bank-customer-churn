import pandas as pd

YES_NO_MAP = {"yes": 1, "no": 0}


def _clean_money_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Strip currency symbols, spaces, and other non-numeric characters
    from a money column and convert to float.
    """
    if col not in df.columns:
        return df

    df = df.copy()
    df[col] = (
        df[col]
        .astype(str)
        # keep digits, dot, minus; drop everything else
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _map_yes_no(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(YES_NO_MAP)
    return df


def _normalize_geography(df: pd.DataFrame, col: str = "Geography") -> pd.DataFrame:
    """
    Normalize Geography values so we only use full country names:
    France, Spain, Germany.

    Handles codes like FRA, FR, ESP, ES, DEU, DE, etc.
    """
    if col not in df.columns:
        return df

    df = df.copy()
    mapping = {
        "fra": "France",
        "fr": "France",
        "france": "France",
        "french": "France",
        "esp": "Spain",
        "es": "Spain",
        "spain": "Spain",
        "deu": "Germany",
        "de": "Germany",
        "ger": "Germany",
        "germany": "Germany",
    }

    df[col] = df[col].astype(str).str.strip().str.lower().map(mapping).fillna(df[col])

    return df


def clean_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the bank churn dataset.

    - Fix Balance & EstimatedSalary formatting (remove currency, commas, etc.)
    - Convert Yes/No to 1/0 for binary columns
    - Drop identifier columns (CustomerId, Surname)
    - Ensure Exited is 0/1 integer
    """
    df = df.copy()

    # Clean money columns
    for col in ["Balance", "EstimatedSalary", "EstimatedSal"]:
        df = _clean_money_column(df, col)

    # If the original short name 'EstimatedSal' existed, rename it
    if "EstimatedSal" in df.columns and "EstimatedSalary" not in df.columns:
        df = df.rename(columns={"EstimatedSal": "EstimatedSalary"})

    # Map Yes/No binary columns
    yes_no_cols = [
        c for c in ["HasCrCard", "IsActiveMember", "ActiveMember"] if c in df.columns
    ]
    df = _map_yes_no(df, yes_no_cols)

    # Normalize Geography
    df = _normalize_geography(df, col="Geography")

    # Drop identifiers
    for col in ["CustomerId", "Surname"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Ensure target is int 0/1
    if "Exited" in df.columns:
        df["Exited"] = (
            pd.to_numeric(df["Exited"], errors="coerce").fillna(0).astype(int)
        )

    print(f"Cleaned dataframe shape: {df.shape}")

    return df
