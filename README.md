![CI](https://github.com/oosei25/bank-customer-churn/actions/workflows/ci.yml/badge.svg)

# Bank Customer Churn – End-to-End Modeling & Explainability

This project simulates a real-world **member churn** use case for a retail bank / credit union.  
Using a sample dataset of bank customers, it builds an end-to-end pipeline to:

- clean and engineer features from messy source files,
- train and compare multiple churn models (Logistic Regression, Random Forest, XGBoost),
- evaluate performance with **AUC** and **lift curves**, and
- explain predictions using **SHAP** at both global and local levels.

The code is organized as a small, reusable Python package (`churn_utils`) plus a single
walk-through notebook.

---

## Project structure

```text
bank-customer-churn/
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions workflow (runs pytest)
├── churn_utils/
│   ├── __init__.py
│   ├── cleaning.py                       # cleaning & type coercion
│   ├── evaluation.py                     # metrics, classification report, lift@k
│   ├── explain.py                        # SHAP-based global & local explanations
│   ├── features.py                       # feature engineering helpers
│   ├── io.py                             # data loading / saving
│   ├── modeling.py                       # train/test split & sklearn pipelines
│   └── viz.py                            # EDA plots (churn by segment, distributions)
├── data/
│   ├── raw/
│   │   ├── Bank_Churn_Data_Dictionary.csv # field descriptions
│   │   └── Bank_Churn_Messy.xlsx          # original messy Excel (two sheets)
│   └── processed/
│       └── bank_churn_clean.csv           # cleaned, model-ready dataset
├── notebooks/
│   ├── 01_churn_eda_and_model.ipynb       # main interactive notebook
│   ├── 01_churn_eda_and_model.md          # Markdown export (full narrative)
│   └── 01_churn_eda_and_model_files/      # images referenced by the .md
├── outputs/
│   └── shap_summary_xgb.png               # example SHAP summary plot
├── tests/
│   ├── conftest.py                        # adds project root to sys.path for tests
│   ├── test_cleaning.py                   # checks cleaning / normalization
│   ├── test_explain.py                    # checks SHAP explainer
│   └── test_modeling.py                   # checks modeling pipeline & XGBoost
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset

The data is based on the public [Bank Customer Churn sample (10,000 customers)](https://mavenanalytics.io/data-playground/bank-customer-churn) with:

- Customer profile – age, gender, geography, estimated salary
- Relationship info – tenure, balance, number of products, credit card flag, active member flag
- Target – Exited (1 = churned, 0 = retained)

> For demonstration purposes the repository also includes a “messy” Excel file
(Bank_Churn_Messy.xlsx) with two sheets (Customer_Info, Account_Info) to showcase
joining, type fixing, and cleaning.

---

## Setup

```bash
git clone <this-repo-url>
cd bank-customer-churn

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Testing

Basic smoke tests are included to validate the core utilities:

- `tests/test_cleaning.py` – checks that the cleaning pipeline returns a
  consistent, numeric schema and normalizes key fields.
- `tests/test_modeling.py` – verifies that the modeling pipeline can split the
  data, fit the XGBoost pipeline, and produce valid churn probabilities.
- `tests/test_explain.py` – verifies that a SHAP explainer can be created for
  the XGBoost model and returns transformed features.

To run the tests:

```bash
pip install -r requirements.txt   # if not already installed
pytest
```

---

## Extending the project

In a production environment, natural next steps would include:

- Adding richer behavioral and time-based features (transaction trends, digital usage).
- Calibrating probabilities for decisioning and capacity planning.
- Exploring ensembles (stacked models) and champion–challenger setups.
- Deploying the model as a service and integrating with campaign tooling.
- Monitoring performance and fairness over time (AUC, lift, calibration, SHAP drift).
