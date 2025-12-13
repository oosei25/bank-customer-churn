import pandas as pd

from churn_utils.modeling import prepare_data_and_pipelines

def _make_tiny_clean_df():
    # Already "clean" schema – similar to bank_churn_clean.csv
    return pd.DataFrame(
        {
            "Age":           [35, 50, 45, 39, 60, 42, 55, 30],
            "Geography":     ["France", "Germany", "Spain", "France",
                              "Germany", "Spain", "France", "Germany"],
            "Gender":        ["Female", "Male", "Female", "Male",
                              "Female", "Male", "Female", "Male"],
            "CreditScore":   [650, 720, 610, 700, 580, 690, 640, 710],
            "Tenure":        [5, 10, 2, 7, 3, 8, 1, 6],
            "Balance":       [10000.0, 0.0, 5000.0, 20000.0,
                              15000.0, 8000.0, 12000.0, 3000.0],
            "NumOfProducts": [1, 2, 1, 3, 2, 1, 2, 1],
            "HasCrCard":     [1, 1, 0, 1, 1, 0, 1, 1],
            "IsActiveMember":[1, 0, 1, 1, 0, 1, 0, 1],
            "EstimatedSalary":[50000.0, 60000.0, 55000.0, 52000.0,
                               58000.0, 61000.0, 54000.0, 59000.0],
            "Exited":        [0, 1, 0, 0, 1, 0, 1, 0],  # at least 3 churners
        }
    )


def test_prepare_pipelines_and_fit_xgb():
    df_clean = _make_tiny_clean_df()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        logreg_pipe,
        rf_pipe,
        xgb_pipe,
    ) = prepare_data_and_pipelines(df_clean)

    # Non-empty splits
    assert len(X_train) > 0
    assert len(X_test) > 0

    # Pipelines fit without error and can predict probabilities
    xgb_pipe.fit(X_train, y_train)
    proba = xgb_pipe.predict_proba(X_test)[:, 1]

    assert len(proba) == len(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()