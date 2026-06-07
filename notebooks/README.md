# Notebooks

Run notebooks in this order:

| Notebook | Purpose |
|---|---|
| `eda-fraud-data.ipynb` | EDA for Fraud_Data.csv — distributions, imbalance, geolocation |
| `eda-creditcard.ipynb` | EDA for creditcard.csv — PCA features, imbalance, correlations |
| `feature-engineering.ipynb` | Time, velocity, and IP-country feature construction |
| `class-imbalance.ipynb` | SMOTE and undersampling demonstration |
| `modeling.ipynb` | *(Task 2)* Logistic Regression, XGBoost, LightGBM training |
| `shap-explainability.ipynb` | *(Task 3)* SHAP summary, force, and dependency plots |

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
jupyter notebook
```

## Notes

- Place raw CSVs in `../data/raw/` before running.
- Notebooks import from `../src/` — ensure the repo root is on your Python path.
- All figures are saved to `../reports/figures/`.
