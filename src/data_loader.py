"""
data_loader.py
==============
Utilities for loading the three raw datasets used in the fraud detection project.

Datasets:
    - Fraud_Data.csv        : E-commerce transaction records with fraud labels.
    - IpAddress_to_Country  : IP range → country mapping for geolocation enrichment.
    - creditcard.csv        : PCA-anonymised bank card transaction records.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_fraud_data(path: Path | None = None) -> pd.DataFrame:
    """Load and return Fraud_Data.csv with typed columns.

    Args:
        path: Override the default raw data path.

    Returns:
        DataFrame with parsed datetime columns and correct dtypes.
    """
    file_path = path or RAW_DIR / "Fraud_Data.csv"
    logger.info("Loading Fraud_Data from %s", file_path)
    df = pd.read_csv(
        file_path,
        parse_dates=["signup_time", "purchase_time"],
        dtype={
            "user_id": "int64",
            "purchase_value": "float64",
            "age": "int64",
            "class": "int8",
        },
    )
    logger.info("Fraud_Data loaded: %d rows, %d cols", *df.shape)
    return df


def load_ip_country(path: Path | None = None) -> pd.DataFrame:
    """Load and return IpAddress_to_Country.csv.

    Args:
        path: Override the default raw data path.

    Returns:
        DataFrame with numeric IP range bounds.
    """
    file_path = path or RAW_DIR / "IpAddress_to_Country.csv"
    logger.info("Loading IP-Country mapping from %s", file_path)
    df = pd.read_csv(
        file_path,
        dtype={
            "lower_bound_ip_address": "float64",
            "upper_bound_ip_address": "float64",
            "country": "str",
        },
    )
    logger.info("IP-Country table loaded: %d rows", len(df))
    return df


def load_creditcard_data(path: Path | None = None) -> pd.DataFrame:
    """Load and return creditcard.csv.

    Args:
        path: Override the default raw data path.

    Returns:
        DataFrame with correct numeric types and 'Class' target column.
    """
    file_path = path or RAW_DIR / "creditcard.csv"
    logger.info("Loading creditcard data from %s", file_path)
    df = pd.read_csv(file_path)
    df["Class"] = df["Class"].astype("int8")
    logger.info("Creditcard data loaded: %d rows, %d cols", *df.shape)
    return df
