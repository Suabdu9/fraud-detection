"""
preprocessor.py
===============
Data cleaning and transformation pipelines for both datasets.

Key responsibilities:
    - Handle missing values (impute or drop with documented justification)
    - Remove duplicate rows
    - Correct data types
    - IP address → integer conversion
    - Range-based merge of Fraud_Data with IP-Country table
    - Normalize / scale numerical features
    - One-hot encode categoricals
    - Apply SMOTE or undersampling on training split ONLY
"""

import logging
import struct
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IP Address Utilities
# ---------------------------------------------------------------------------

def ip_to_int(ip_str: str) -> int:
    """Convert a dotted-decimal IPv4 string to a 32-bit integer.

    Args:
        ip_str: e.g. "192.168.1.1"

    Returns:
        Integer representation of the IP address.
    """
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip_str)))[0]
    except (OSError, ValueError):
        return np.nan


def merge_ip_country(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich e-commerce transactions with country via range-based IP lookup.

    The IpAddress_to_Country table stores IP ranges as floats.  We convert
    each transaction's IP to an integer and find the matching range using a
    sorted merge + searchsorted approach — O(n log n) rather than O(n*m).

    Args:
        fraud_df:      Fraud_Data DataFrame (must have 'ip_address' column).
        ip_country_df: IP-Country mapping DataFrame.

    Returns:
        fraud_df with an added 'country' column.  Unmatched rows get NaN.
    """
    logger.info("Converting IP addresses to integers …")
    fraud_df = fraud_df.copy()
    fraud_df["ip_int"] = fraud_df["ip_address"].apply(ip_to_int)

    # Sort both tables for binary search
    ip_country_sorted = ip_country_df.sort_values("lower_bound_ip_address").reset_index(drop=True)
    lower_bounds = ip_country_sorted["lower_bound_ip_address"].values
    upper_bounds = ip_country_sorted["upper_bound_ip_address"].values
    countries = ip_country_sorted["country"].values

    def lookup(ip_int):
        if np.isnan(ip_int):
            return np.nan
        idx = np.searchsorted(lower_bounds, ip_int, side="right") - 1
        if idx >= 0 and ip_int <= upper_bounds[idx]:
            return countries[idx]
        return np.nan

    fraud_df["country"] = fraud_df["ip_int"].apply(lookup)
    matched = fraud_df["country"].notna().sum()
    logger.info("IP-Country merge complete: %d / %d rows matched", matched, len(fraud_df))
    return fraud_df


# ---------------------------------------------------------------------------
# Fraud_Data Pipeline
# ---------------------------------------------------------------------------

def preprocess_fraud_data(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Full preprocessing pipeline for Fraud_Data.csv.

    Steps:
        1. Drop duplicates
        2. Handle missing values
        3. Type corrections
        4. IP → country merge
        5. Feature engineering (delegated to feature_engineering.py)
        6. One-hot encode categoricals
        7. Scale numerics

    Args:
        fraud_df:      Raw Fraud_Data DataFrame.
        ip_country_df: Raw IP-Country DataFrame.
        scaler:        Pre-fitted StandardScaler (for test set transform).
        fit_scaler:    If True, fit a new scaler (use on training set only).

    Returns:
        Tuple of (preprocessed DataFrame, fitted StandardScaler).
    """
    df = fraud_df.copy()
    logger.info("=== Preprocessing Fraud_Data ===")

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Duplicates removed: %d rows dropped", before - len(df))

    # 2. Missing values
    # - 'browser', 'source', 'sex' : categorical → fill with 'Unknown'
    # - 'age' : numeric  → median imputation (robust to outliers)
    # - 'ip_address' : required for geolocation; rows without it are dropped
    cat_cols = ["browser", "source", "sex"]
    for col in cat_cols:
        n_missing = df[col].isna().sum()
        if n_missing:
            df[col] = df[col].fillna("Unknown")
            logger.info("  '%s': %d missing → filled with 'Unknown'", col, n_missing)

    if df["age"].isna().any():
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age)
        logger.info("  'age': missing values → imputed with median %.1f", median_age)

    before = len(df)
    df = df.dropna(subset=["ip_address"])
    logger.info("  Rows dropped (missing ip_address): %d", before - len(df))

    # 3. Type corrections
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])

    # 4. IP-Country merge
    df = merge_ip_country(df, ip_country_df)

    # 5. Feature engineering (time-based + velocity)
    from src.feature_engineering import engineer_features
    df = engineer_features(df)

    # 6. One-hot encode categoricals
    cat_encode = ["source", "browser", "sex", "country"]
    # Keep top N categories for country to avoid high cardinality explosion
    top_countries = df["country"].value_counts().nlargest(20).index.tolist()
    df["country"] = df["country"].where(df["country"].isin(top_countries), other="Other")
    df = pd.get_dummies(df, columns=cat_encode, drop_first=False, dtype=int)
    logger.info("One-hot encoding applied to: %s", cat_encode)

    # 7. Scale numerics
    num_cols = ["purchase_value", "age", "time_since_signup_hours", "hour_of_day",
                "day_of_week", "transaction_count_1h", "transaction_count_24h"]
    num_cols = [c for c in num_cols if c in df.columns]  # guard against missing

    if fit_scaler:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logger.info("StandardScaler fitted and applied to numeric features")
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        logger.info("Pre-fitted StandardScaler applied to numeric features")

    logger.info("Preprocessing complete. Shape: %s", df.shape)
    return df, scaler


# ---------------------------------------------------------------------------
# Creditcard Pipeline
# ---------------------------------------------------------------------------

def preprocess_creditcard_data(
    cc_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Full preprocessing pipeline for creditcard.csv.

    Steps:
        1. Drop duplicates
        2. Handle missing values (PCA features are already anonymised)
        3. Scale 'Time' and 'Amount' (V1-V28 already unit-scaled by PCA)

    Args:
        cc_df:      Raw creditcard DataFrame.
        scaler:     Pre-fitted StandardScaler for test transform.
        fit_scaler: If True, fit new scaler (training set only).

    Returns:
        Tuple of (preprocessed DataFrame, fitted StandardScaler).
    """
    df = cc_df.copy()
    logger.info("=== Preprocessing creditcard data ===")

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Duplicates removed: %d rows dropped", before - len(df))

    # 2. Missing values — PCA features should have none; verify and drop if any
    missing = df.isna().sum().sum()
    if missing:
        logger.warning("Found %d missing values in creditcard data — dropping affected rows", missing)
        df = df.dropna()
    else:
        logger.info("No missing values found in creditcard data")

    # 3. Scale 'Time' and 'Amount' only
    scale_cols = ["Time", "Amount"]
    if fit_scaler:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        logger.info("StandardScaler fitted on Time and Amount")
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])
        logger.info("Pre-fitted scaler applied to Time and Amount")

    logger.info("Preprocessing complete. Shape: %s", df.shape)
    return df, scaler


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to the training set.

    Justification: SMOTE generates synthetic minority samples in feature space,
    preserving more information than pure undersampling.  Applied ONLY to the
    training split to prevent data leakage.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Resampled (X_train, y_train) with balanced class distribution.
    """
    logger.info("Applying SMOTE. Before: %s", y_train.value_counts().to_dict())
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info("After SMOTE: %s", pd.Series(y_res).value_counts().to_dict())
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def apply_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply random undersampling to the training set.

    Useful when the dataset is very large and SMOTE is computationally expensive.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        sampling_strategy: Ratio of minority to majority after resampling.
        random_state: Seed for reproducibility.

    Returns:
        Resampled (X_train, y_train).
    """
    logger.info("Applying RandomUnderSampler. Before: %s", y_train.value_counts().to_dict())
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    logger.info("After undersampling: %s", pd.Series(y_res).value_counts().to_dict())
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)
