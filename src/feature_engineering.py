"""
feature_engineering.py
======================
Behavioural, temporal, and geolocation feature engineering for Fraud_Data.

Features created:
    - time_since_signup_hours : time elapsed from account creation to purchase
    - hour_of_day             : hour of purchase (captures time-of-day fraud patterns)
    - day_of_week             : day of purchase (0=Monday … 6=Sunday)
    - transaction_count_1h    : number of transactions by this user in the past 1 hour
    - transaction_count_24h   : number of transactions by this user in the past 24 hours

Rationale:
    Fraudsters often:
      * Purchase immediately after creating a new account.
      * Cluster multiple transactions in a short burst before detection.
      * Prefer off-hours (late night / early morning) when monitoring is lighter.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _transaction_velocity(df: pd.DataFrame, window_hours: int) -> pd.Series:
    """Count prior transactions per user within a rolling time window.

    Uses a sort + groupby approach to avoid an O(n^2) loop.

    Args:
        df:           DataFrame with 'user_id' and 'purchase_time' columns.
        window_hours: Rolling window size in hours.

    Returns:
        Series of transaction counts aligned to df.index.
    """
    df_sorted = df[["user_id", "purchase_time"]].copy().sort_values("purchase_time")
    window = pd.Timedelta(hours=window_hours)

    counts = []
    for _, group in df_sorted.groupby("user_id"):
        times = group["purchase_time"].values
        cnt = []
        for i, t in enumerate(times):
            # count earlier transactions within window (exclusive of current)
            cnt.append(((times[:i] >= (t - window.to_timedelta64())) & (times[:i] < t)).sum())
        counts.extend(cnt)

    result = pd.Series(counts, index=df_sorted.index, name=f"transaction_count_{window_hours}h")
    return result.reindex(df.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to Fraud_Data.

    Args:
        df: DataFrame that has passed through data cleaning (must have
            'signup_time', 'purchase_time', 'user_id' columns).

    Returns:
        DataFrame with new engineered columns appended.
    """
    df = df.copy()
    logger.info("Starting feature engineering …")

    # -- Temporal features ------------------------------------------------
    df["time_since_signup_hours"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600
    # Clamp negative values (data quality: purchase before signup → set to 0)
    negative_mask = df["time_since_signup_hours"] < 0
    if negative_mask.any():
        logger.warning(
            "%d rows have purchase_time < signup_time — clamping to 0",
            negative_mask.sum(),
        )
        df.loc[negative_mask, "time_since_signup_hours"] = 0.0

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    logger.info("Temporal features created: time_since_signup_hours, hour_of_day, day_of_week")

    # -- Transaction velocity features ------------------------------------
    df["transaction_count_1h"] = _transaction_velocity(df, window_hours=1)
    df["transaction_count_24h"] = _transaction_velocity(df, window_hours=24)

    logger.info(
        "Velocity features created: transaction_count_1h (mean=%.2f), "
        "transaction_count_24h (mean=%.2f)",
        df["transaction_count_1h"].mean(),
        df["transaction_count_24h"].mean(),
    )

    # -- Drop raw datetime columns (no longer needed after feature extraction) --
    df = df.drop(columns=["signup_time", "purchase_time", "ip_address", "ip_int",
                           "user_id", "device_id"], errors="ignore")

    logger.info("Feature engineering complete. New shape: %s", df.shape)
    return df
