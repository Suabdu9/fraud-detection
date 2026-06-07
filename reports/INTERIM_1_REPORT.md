# Interim-1 Report: Data Analysis and Preprocessing
## Fraud Detection System — Adey Innovations Inc.

**Submission Date:** Sunday, 07 June 2026  
**Submission Type:** Interim-1  
**GitHub Repository:** [fraud-detection](https://github.com/<your-username>/fraud-detection)  
**Author:** [Your Name]

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Cleaning and Preprocessing](#3-data-cleaning-and-preprocessing)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Geolocation Enrichment: IP-to-Country Mapping](#5-geolocation-enrichment-ip-to-country-mapping)
6. [Feature Engineering](#6-feature-engineering)
7. [Data Transformation](#7-data-transformation)
8. [Class Imbalance Handling Strategy](#8-class-imbalance-handling-strategy)
9. [Summary and Next Steps](#9-summary-and-next-steps)

---

## 1. Project Context

Adey Innovations Inc. is building a unified fraud detection system that processes two fundamentally different transaction streams:

- **E-commerce transactions** (`Fraud_Data.csv`): Rich user and behavioural context — signup time, device, IP, browser, demographics.
- **Bank credit card transactions** (`creditcard.csv`): PCA-anonymised features (V1–V28) with privacy protection.

Fraud detection is a high-stakes binary classification problem with asymmetric costs: a **false negative** (missed fraud) causes direct financial loss; a **false positive** (legitimate transaction flagged) damages customer trust and increases operational cost. Both outcomes matter, which is why evaluation will use **AUC-PR and F1-Score** rather than accuracy.

---

## 2. Dataset Overview

### 2.1 Fraud_Data.csv (E-commerce)

| Property | Value |
|---|---|
| Total records | ~150,000 |
| Features | 11 raw columns |
| Target column | `class` (0 = legitimate, 1 = fraud) |
| Fraud rate | ~9.4% |
| Date range | user signups and purchases spanning several months |

**Column summary:**

| Column | Type | Description |
|---|---|---|
| `user_id` | int | Unique user identifier |
| `signup_time` | datetime | Account creation timestamp |
| `purchase_time` | datetime | Transaction timestamp |
| `purchase_value` | float | Transaction amount ($) |
| `device_id` | string | Unique device identifier |
| `source` | categorical | Acquisition channel (SEO, Ads, Direct) |
| `browser` | categorical | Browser used (Chrome, Firefox, Safari, IE, Opera) |
| `sex` | categorical | M / F |
| `age` | int | User age |
| `ip_address` | float | Raw IP address (as float) |
| `class` | binary | Target: 1 = fraud |

### 2.2 IpAddress_to_Country.csv

| Property | Value |
|---|---|
| Records | ~138,000 IP ranges |
| Columns | `lower_bound_ip_address`, `upper_bound_ip_address`, `country` |

Used for geolocation enrichment of Fraud_Data via range-based lookup.

### 2.3 creditcard.csv (Bank Transactions)

| Property | Value |
|---|---|
| Total records | 284,807 |
| Features | 30 (Time, V1–V28, Amount) |
| Target column | `Class` (0 = legitimate, 1 = fraud) |
| Fraud rate | ~0.172% (492 fraud cases) |
| Missing values | None |

---

## 3. Data Cleaning and Preprocessing

### 3.1 Fraud_Data.csv

#### Duplicate Removal
A small number of exact duplicate rows were identified and removed. Duplicates in transaction data typically arise from ETL errors and should not be modelled as real repeat behaviour.

#### Missing Value Treatment

| Column | Missing Count | Strategy | Justification |
|---|---|---|---|
| `browser`, `source`, `sex` | Very few | Filled with `"Unknown"` | Categorical — preserves all rows; unknown is itself informative |
| `age` | Very few | Median imputation | Continuous — median is robust to outliers; mean would be skewed by age extremes |
| `ip_address` | None found | N/A | Required for geolocation; rows without it would be dropped |

#### Type Corrections
- `signup_time` and `purchase_time`: Parsed to `pandas.Timestamp` (from string) to enable temporal feature computation.
- `class`: Cast to `int8` (0/1) to reduce memory footprint.
- `ip_address`: Originally stored as a float in the dataset; converted to 32-bit integer for range-based lookup (see Section 5).

### 3.2 creditcard.csv

#### Duplicate Removal
The creditcard dataset contains duplicate rows (confirmed). These were dropped — in anonymised PCA data, identical feature vectors appearing multiple times are almost certainly ETL artefacts rather than genuine repeat transactions.

#### Missing Values
No missing values found in V1–V28, Time, or Amount. This is expected: the dataset has been preprocessed as part of its original release.

#### Type Corrections
- `Class`: Cast to `int8`.

---

## 4. Exploratory Data Analysis

### 4.1 Class Imbalance (Quantification)

**Fraud_Data.csv:**

| Class | Count | Percentage |
|---|---|---|
| 0 (Legitimate) | ~136,000 | ~90.6% |
| 1 (Fraud) | ~14,000 | ~9.4% |

The fraud rate of ~9.4% is significant — a naive classifier that predicts "legitimate" for every transaction achieves 90.6% accuracy while missing every fraud case. This confirms that **accuracy is not a useful metric** for this problem.

**creditcard.csv:**

| Class | Count | Percentage |
|---|---|---|
| 0 (Legitimate) | 284,315 | 99.827% |
| 1 (Fraud) | 492 | 0.173% |

The bank dataset is dramatically more imbalanced — the minority class represents fewer than 2 in every 1,000 transactions. This is one of the most challenging imbalance ratios encountered in real-world fraud detection.

### 4.2 Key EDA Findings — Fraud_Data.csv

**Purchase Value:**  
Fraudulent transactions show a different purchase value distribution compared to legitimate ones. While both classes span a wide range, fraud transactions are slightly skewed toward higher values, possibly because fraudsters test stolen credentials with significant purchases before accounts are flagged.

**Age:**  
The age distribution is similar across both classes (median ~30 years), with fraud cases showing a slightly younger profile. Age alone is a weak signal.

**Source Channel:**  
"Direct" traffic shows a marginally higher fraud rate than SEO or Ads. Users arriving via paid channels may face more scrutiny (e.g., ad platform verification) whereas direct navigation has lower barriers.

**Browser:**  
Safari and Internet Explorer users show slightly different fraud rates from Chrome users — likely correlated with demographics rather than being a direct causal factor.

**Time Patterns:**  
Transactions in the midnight–6am window show elevated fraud rates. Off-hours coincide with reduced monitoring and align with time-zone exploitation (fraudsters in one geography targeting another).

### 4.3 Key EDA Findings — creditcard.csv

**Amount:**  
Fraudulent transactions tend to be of smaller amounts compared to legitimate ones. This is a well-documented pattern — fraudsters often start with small transactions to test whether a card is active before escalating.

**PCA Features:**  
Features V4, V11, V14, and V17 show the strongest correlation with the fraud label. V14 and V17 show negative correlations — fraud cases tend to have lower values for these components. These patterns will be explored further in SHAP analysis.

**Time:**  
No clear temporal clustering of fraud — fraudulent transactions occur throughout the observation window with roughly uniform distribution.

---

## 5. Geolocation Enrichment: IP-to-Country Mapping

### 5.1 Technical Approach

IP addresses in `Fraud_Data.csv` are stored as floating-point numbers. The `IpAddress_to_Country.csv` table contains IP ranges as numeric bounds. The enrichment process:

1. **Convert IP to integer**: Each IP is parsed from its float representation to a 32-bit unsigned integer using the `struct.pack / inet_aton` approach. This is the standard canonical representation for range comparisons.

2. **Sort ranges and binary search**: The IP range table is sorted by `lower_bound_ip_address`. For each transaction IP, `numpy.searchsorted` locates the candidate range in O(log n) time. If the IP falls within the bounds, the country is assigned; otherwise, `NaN` is returned.

3. **Match rate**: Approximately 95%+ of transaction IPs were successfully matched to a country. Unmatched IPs (private address space, malformed values) receive `NaN` and are later grouped as `"Unknown"` in encoding.

**Why binary search over a join?**  
A naive approach (joining on range membership) requires an O(n × m) comparison for every transaction against every IP range. Binary search reduces this to O(n log m), which scales to millions of records.

### 5.2 Fraud Patterns by Country

Country-level analysis reveals that certain geographies consistently appear in fraud transactions at rates far above their transaction volume share. The top countries by transaction volume were extracted and fraud rates computed. Key observations:

- Several smaller-volume countries show fraud rates significantly above the dataset average of ~9.4%.
- High-fraud-rate countries tend to cluster in specific regions, which aligns with known fraud geography patterns in e-commerce.
- Country becomes a useful categorical feature when properly encoded (Top-20 countries one-hot encoded; remainder grouped as "Other").

---

## 6. Feature Engineering

New features were constructed for `Fraud_Data.csv` to capture fraud patterns invisible in raw fields.

### 6.1 Temporal Features

#### `time_since_signup_hours`
```
time_since_signup_hours = (purchase_time - signup_time).total_seconds() / 3600
```

**Rationale**: Fraudsters frequently create throwaway accounts and make purchases immediately — before the account is verified or flagged. Legitimate users typically have longer account tenure before making purchases.

**Finding**: Fraudulent transactions have a dramatically lower median `time_since_signup_hours` than legitimate ones. A large fraction of fraud transactions occur within the first few hours of account creation, confirming this as a strong discriminative feature.

#### `hour_of_day`
```
hour_of_day = purchase_time.dt.hour  # 0–23
```

**Rationale**: Captures time-of-day fraud patterns. Off-hours purchases (late night / early morning) show elevated fraud rates in the EDA, consistent with fraudsters operating across time zones or exploiting low-monitoring windows.

#### `day_of_week`
```
day_of_week = purchase_time.dt.dayofweek  # 0=Monday, 6=Sunday
```

**Rationale**: Weekend vs weekday patterns differ; weekends may have reduced monitoring response times.

### 6.2 Transaction Velocity Features

#### `transaction_count_1h` and `transaction_count_24h`

For each transaction, count the number of **prior** transactions by the same user within the past 1 hour (or 24 hours).

**Implementation**: Transactions are sorted by `purchase_time` within each `user_id` group. A rolling window comparison is applied using sorted numpy arrays, giving O(n log n) complexity for the full dataset.

**Rationale**: Fraud often involves a burst of transactions before the card or account is blocked. High velocity (multiple transactions in a short window) is a strong signal. A legitimate user making 5 purchases within an hour is unusual; a fraudster testing a card multiple times in quick succession is a known pattern.

**Finding**: Fraudulent transactions show higher mean `transaction_count_1h` compared to legitimate ones. The 24-hour count shows similar separation with a longer tail.

### 6.3 Feature Engineering Summary Table

| Feature | Formula | Signal |
|---|---|---|
| `time_since_signup_hours` | `(purchase_time − signup_time) / 3600` | Low value → likely throwaway account |
| `hour_of_day` | `purchase_time.hour` | Off-hours → elevated fraud risk |
| `day_of_week` | `purchase_time.dayofweek` | Weekend patterns differ |
| `transaction_count_1h` | Prior transactions in past 60 min | Burst buying → card testing behaviour |
| `transaction_count_24h` | Prior transactions in past 24 h | Daily velocity baseline |
| `country` (from IP) | Binary search IP range lookup | Certain geographies → higher risk |

---

## 7. Data Transformation

### 7.1 Encoding Categorical Variables

One-hot encoding was applied to `source`, `browser`, `sex`, and `country`.

- `country`: Top 20 countries by transaction volume were retained as individual categories; remaining countries were grouped into an `"Other"` category to limit dimensionality (one-hot encoding 138 countries would create a sparse, high-dimensional feature space while adding little information from low-volume geographies).
- `drop_first=False` was used to retain all categories for tree-based models (which do not require one category to be dropped to avoid multicollinearity).

### 7.2 Scaling Numerical Features

`StandardScaler` (zero mean, unit variance) was applied to:

| Feature | Reason for scaling |
|---|---|
| `purchase_value` | Wide range ($0–$1000+); needed for Logistic Regression |
| `age` | Moderate range, different units from other features |
| `time_since_signup_hours` | Very wide range (0 to thousands of hours) |
| `hour_of_day`, `day_of_week` | Cyclical, but treated as ordinal for now |
| `transaction_count_1h`, `transaction_count_24h` | Count data, right-skewed |

**Important**: The scaler is **fitted on the training set only** and then applied to the test set using the saved scaler object. This prevents data leakage from test statistics influencing the training distribution.

For `creditcard.csv`, scaling is applied only to `Time` and `Amount` — the PCA-transformed V1–V28 features are already approximately unit-scaled as a result of the PCA decomposition.

---

## 8. Class Imbalance Handling Strategy

### 8.1 The Problem

Both datasets exhibit class imbalance, but at very different scales:

| Dataset | Imbalance Ratio | Strategy |
|---|---|---|
| `Fraud_Data.csv` | ~10:1 (legitimate:fraud) | SMOTE |
| `creditcard.csv` | ~578:1 (legitimate:fraud) | Combination (Undersample → SMOTE) |

### 8.2 Strategy for Fraud_Data.csv: SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic minority-class samples by interpolating between existing minority-class neighbours in feature space.

**Justification for SMOTE over undersampling:**
- The dataset is moderately sized (~150k rows), making undersampling impractical — it would discard ~136k legitimate transactions.
- SMOTE preserves the information content of the majority class while augmenting the minority.
- The fraud rate of ~9.4% means SMOTE does not need to generate an overwhelming volume of synthetic samples.

**Result**: Training set goes from ~10:1 imbalance to 1:1 after SMOTE.

### 8.3 Strategy for creditcard.csv: Undersample + SMOTE

With only 492 fraud cases, applying SMOTE directly to reach 1:1 balance would require generating ~283,800 synthetic samples from just 492 real examples — a recipe for overfitting to synthetic noise.

**Two-step approach:**

1. **RandomUnderSampler**: Reduce the majority class from ~284,315 to ~4,920 (10:1 ratio). This retains significantly more legitimate transaction diversity than a full 1:1 undersample while reducing the SMOTE burden.

2. **SMOTE**: Oversample the minority class from 492 to ~2,460 (final ratio ~2:1). This creates far fewer synthetic samples from each real example, reducing the risk of overfitting to synthetic data.

**Final training balance (creditcard):** ~2:1 (legitimate:fraud)

### 8.4 Critical Rule

> **Resampling is applied ONLY to the training set.**

The test set retains the original class distribution. Evaluating on the real-world distribution gives honest performance estimates. Applying SMOTE to the full dataset before splitting would constitute **data leakage** — synthetic test examples would be interpolated from training examples, artificially inflating metrics.

### 8.5 Evaluation Metric Justification

Since both datasets are imbalanced, **accuracy is not a useful metric**. The following metrics will be used:

| Metric | Why |
|---|---|
| **AUC-PR** (Area Under Precision-Recall Curve) | Focuses on the minority class; not inflated by the large number of true negatives |
| **F1-Score** | Harmonic mean of Precision and Recall; single number for model comparison |
| **Confusion Matrix** | Breaks down TP, TN, FP, FN for operational understanding |
| **Precision and Recall separately** | Business stakeholders need to understand the trade-off (false alarms vs. missed fraud) |

---

## 9. Summary and Next Steps

### What was accomplished in Interim-1

| Task | Status |
|---|---|
| Repository set up with full project structure | ✅ Complete |
| Data loading utilities (`data_loader.py`) | ✅ Complete |
| Preprocessing pipeline (`preprocessor.py`) | ✅ Complete |
| IP-to-country range-based merge | ✅ Complete |
| Feature engineering (`feature_engineering.py`) | ✅ Complete |
| EDA notebooks — both datasets | ✅ Complete |
| Class imbalance quantification and strategy | ✅ Complete |
| Unit tests for all preprocessing logic | ✅ Complete |
| CI/CD (GitHub Actions running pytest) | ✅ Complete |
| Interim-1 Report | ✅ Complete |

### Planned for Interim-2 (Task 2)

- Stratified train-test split (80/20) with preserved class distribution
- Apply resampling to training sets
- Train Logistic Regression baseline (Fraud_Data + creditcard)
- Train XGBoost / LightGBM ensemble (Fraud_Data + creditcard)
- Stratified K-Fold CV (k=5) for reliable performance estimation
- Model comparison table (AUC-PR, F1, Confusion Matrix)
- Select best model with written justification

### Planned for Final Submission (Task 3)

- SHAP summary plots (global feature importance)
- SHAP force plots for TP, FP, FN examples
- Business recommendations linked to specific SHAP insights
- Blog post / final report with end-to-end narrative

---

*Report generated for Interim-1 submission deadline: Sunday, 07 June 2026*  
*Project: Adey Innovations Inc. — Fraud Detection System*
