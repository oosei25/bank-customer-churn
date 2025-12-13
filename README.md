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
├── churn_utils/
│   ├── __init__.py
│   ├── io.py             # data loading / saving
│   ├── cleaning.py       # cleaning & type coercion
│   ├── features.py       # feature engineering helpers
│   ├── modeling.py       # train/test split & sklearn pipelines
│   ├── evaluation.py     # metrics, classification report, lift@k
│   ├── viz.py            # EDA plots (churn by segment, distributions)
│   └── explain.py        # SHAP-based global & local explanations
├── data/
│   ├── raw/
│   │   ├── Bank_Churn_Messy.xlsx        # original messy source (multi-sheet)
│   │   └── bank_churn_raw.csv           # flat raw export
│   └── processed/
│       └── bank_churn_clean.csv         # cleaned, model-ready dataset
├── notebooks/
│   └── 01_churn_eda_and_model.ipynb     # main analysis notebook
├── outputs/
│   └── shap_summary_xgb.png             # example explainability plot
├── requirements.txt
└── README.md
```

---

## Dataset

The data is based on the public [Bank Customer Churn sample (10,000 customers)](https://mavenanalytics.io/data-playground/bank-customer-churn?utm_source=chatgpt.com) with:

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

## Extending the project

In a production environment, natural next steps would include:

- Adding richer behavioral and time-based features (transaction trends, digital usage).
- Calibrating probabilities for decisioning and capacity planning.
- Exploring ensembles (stacked models) and champion–challenger setups.
- Deploying the model as a service and integrating with campaign tooling.
- Monitoring performance and fairness over time (AUC, lift, calibration, SHAP drift).