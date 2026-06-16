# Fraud Detection — Adey Innovations Inc.

End-to-end machine learning system for detecting fraud across e-commerce transactions and bank credit card transactions.

## Project Overview

| Dataset | Records | Fraud Rate | Key Challenge |
|---|---|---|---|
| `Fraud_Data.csv` | ~150,000 | ~9.4% | Behavioural, device, geo signals |
| `creditcard.csv` | 284,807 | ~0.17% | Extreme imbalance, PCA-anonymised |

**Best models:** LightGBM (Fraud_Data, AUC-PR ~0.94) · XGBoost (creditcard, AUC-PR ~0.87)

## Repository Structure

```
fraud-detection/
├── .github/workflows/
│   └── unittests.yml        # CI/CD — runs pytest on every push
├── .vscode/
│   └── settings.json        # Editor config (formatter, linter)
├── data/
│   ├── raw/                 # Original CSVs — place datasets here (gitignored)
│   └── processed/           # Cleaned & feature-engineered outputs (gitignored)
├── models/                  # Saved model artifacts (.joblib) — output of modeling.ipynb
├── notebooks/
│   ├── eda-fraud-data.ipynb         # Task 1 — EDA for Fraud_Data
│   ├── eda-creditcard.ipynb         # Task 1 — EDA for creditcard
│   ├── feature-engineering.ipynb    # Task 1 — Feature construction & analysis
│   ├── class-imbalance.ipynb        # Task 1 — SMOTE vs undersampling demo
│   ├── modeling.ipynb               # Task 2 — Train, evaluate, compare models
│   └── shap-explainability.ipynb    # Task 3 — SHAP plots & business insights
├── src/
│   ├── data_loader.py        # Load all 3 datasets with correct dtypes
│   ├── preprocessor.py       # Cleaning, IP merge, scaling, SMOTE
│   ├── feature_engineering.py # time_since_signup, velocity, temporal features
│   ├── trainer.py            # Model definitions, training, CV, evaluation
│   └── explainer.py          # SHAP utilities and force plot helpers
├── tests/
│   └── test_preprocessing.py # Unit tests — IP conversion, features, resampling
├── reports/
│   ├── INTERIM_1_REPORT.md   # Task 1 submission report
│   ├── INTERIM_2_REPORT.md   # Task 2 submission report
│   ├── FINAL_REPORT.md       # Complete end-to-end project report
│   └── figures/              # Plots saved by notebooks
├── scripts/                  # Standalone pipeline scripts
├── requirements.txt          # Python dependencies
├── setup_repo.sh             # One-command environment setup
└── .gitignore
```

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
bash setup_repo.sh
```

### 2. Add raw datasets

Place the three CSV files in `data/raw/`:

```
data/raw/
├── Fraud_Data.csv
├── IpAddress_to_Country.csv
└── creditcard.csv
```

### 3. Run notebooks in order

```bash
source venv/bin/activate
jupyter notebook notebooks/
```

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `eda-fraud-data.ipynb` | Explore Fraud_Data — distributions, imbalance, geo |
| 2 | `eda-creditcard.ipynb` | Explore creditcard — PCA features, correlations |
| 3 | `feature-engineering.ipynb` | Build & analyse engineered features |
| 4 | `class-imbalance.ipynb` | Demonstrate resampling strategies |
| 5 | `modeling.ipynb` | Train all models, compare, save best |
| 6 | `shap-explainability.ipynb` | SHAP analysis + business recommendations |

### 4. Run tests

```bash
pytest tests/ -v
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **AUC-PR over accuracy** | Accuracy is misleading on imbalanced data; AUC-PR focuses on the minority fraud class |
| **SMOTE training-set only** | Prevents data leakage — test set must reflect real-world distribution |
| **Binary search for IP lookup** | O(n log m) vs O(n×m) naive join — scales to millions of records |
| **Top-20 country encoding** | Limits one-hot dimensionality while retaining the most informative geographies |
| **LightGBM for Fraud_Data** | Best AUC-PR; faster training; handles high-cardinality categoricals well |
| **XGBoost for creditcard** | `scale_pos_weight` directly encodes 578:1 class ratio into loss function |

## Results Summary

### Fraud_Data.csv

| Model | F1 | AUC-PR |
|---|---|---|
| Logistic Regression | ~0.72 | ~0.78 |
| Random Forest | ~0.88 | ~0.92 |
| XGBoost | ~0.89 | ~0.93 |
| **LightGBM (best)** | **~0.90** | **~0.94** |

### creditcard.csv

| Model | F1 | AUC-PR |
|---|---|---|
| Logistic Regression | ~0.71 | ~0.73 |
| LightGBM | ~0.85 | ~0.86 |
| **XGBoost (best)** | **~0.86** | **~0.87** |

## Top Business Recommendations (from SHAP)

1. **Step-up auth** for purchases within 2 hours of account creation
2. **Rate-limit** accounts making >2 transactions in 60 minutes
3. **Geo-risk scoring** — extra friction from high-fraud-rate countries
4. **Off-hours + high-value alert** — flag midnight–6am transactions > $200
5. **Aged-account fraud layer** — separate model for accounts > 30 days old

## Team

- **Organisation:** Adey Innovations Inc.
- **Tutors:** Kerod, Mahbubah, Feven
- **Slack:** #all-week5-and-6
