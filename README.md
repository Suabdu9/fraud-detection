# Fraud Detection — Adey Innovations Inc.

End-to-end machine learning system for detecting fraud across e-commerce transactions and bank credit card transactions.

## Project Overview

Adey Innovations is building a unified fraud detection capability across two very different data streams:

| Dataset | Records | Features | Fraud Rate |
|---|---|---|---|
| `Fraud_Data.csv` | ~150,000 | Behavioural, device, geo | ~9% |
| `creditcard.csv` | 284,807 | PCA-anonymised (V1–V28) | ~0.17% |

Both problems are characterised by **severe class imbalance**, which shapes every choice from evaluation metrics (AUC-PR, F1) to resampling strategy (SMOTE, undersampling).

## Repository Structure

```
fraud-detection/
├── .vscode/settings.json          # Editor configuration
├── .github/workflows/
│   └── unittests.yml              # CI/CD — runs pytest on push
├── data/                          # ← in .gitignore, never committed
│   ├── raw/                       # Original CSVs go here
│   └── processed/                 # Cleaned / feature-engineered outputs
├── notebooks/
│   ├── eda-fraud-data.ipynb       # Task 1 — EDA for Fraud_Data
│   ├── eda-creditcard.ipynb       # Task 1 — EDA for creditcard
│   ├── feature-engineering.ipynb  # Task 1 — Feature construction
│   ├── class-imbalance.ipynb      # Task 1 — SMOTE vs undersampling
│   ├── modeling.ipynb             # Task 2 — Model training (coming)
│   ├── shap-explainability.ipynb  # Task 3 — SHAP analysis (coming)
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading utilities
│   ├── preprocessor.py            # Cleaning, IP merge, scaling, resampling
│   └── feature_engineering.py    # Temporal + velocity feature construction
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py     # Unit tests for all preprocessing logic
├── models/                        # Saved model artifacts (.pkl / .joblib)
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add raw data

Download the three datasets and place them in `data/raw/`:

```
data/raw/
├── Fraud_Data.csv
├── IpAddress_to_Country.csv
└── creditcard.csv
```

### 5. Run unit tests

```bash
pytest tests/ -v
```

### 6. Launch Jupyter

```bash
jupyter notebook notebooks/
```

## Pipeline Summary

### Task 1 — Data Analysis & Preprocessing *(Interim-1)*

- **EDA**: Univariate/bivariate distributions, class imbalance quantification
- **Geolocation**: IP → integer → range lookup → country assignment
- **Feature Engineering**: `time_since_signup_hours`, `hour_of_day`, `day_of_week`, `transaction_count_1h`, `transaction_count_24h`
- **Resampling**: SMOTE for Fraud_Data; Undersampling + SMOTE for creditcard

### Task 2 — Model Building *(Interim-2)*

- Logistic Regression baseline
- XGBoost / LightGBM ensemble
- Stratified K-Fold CV (k=5)
- Metrics: AUC-PR, F1, Confusion Matrix

### Task 3 — Explainability *(Final)*

- SHAP summary plots (global importance)
- SHAP force plots (individual predictions: TP, FP, FN)
- Business recommendations mapped to SHAP insights

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **AUC-PR over AUC-ROC** | AUC-ROC is misleading on highly imbalanced data; Precision-Recall focuses on the minority class |
| **SMOTE on training set only** | Prevents data leakage — test distribution must reflect real world |
| **StandardScaler** | Logistic Regression requires scaled features; tree models are robust to scaling but we apply it uniformly for consistency |
| **Top-20 country encoding** | Limits one-hot cardinality explosion while preserving the most informative geography signals |
| **Range-based IP lookup** | O(n log n) binary search via `searchsorted` — scales to millions of records |

## Team

- **Organization**: Adey Innovations Inc.
- **Role**: Data Scientist
- **Tutors**: Kerod, Mahbubah, Feven
- **Slack**: #all-week5-and-6

## License

Internal project — Adey Innovations Inc.
