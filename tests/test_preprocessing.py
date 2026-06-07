"""
tests/test_preprocessing.py
============================
Unit tests for data loading, IP conversion, feature engineering, and resampling utilities.
"""

import struct
import socket

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import ip_to_int, merge_ip_country, apply_smote, apply_undersampling
from src.feature_engineering import engineer_features


# ─── ip_to_int ──────────────────────────────────────────────────────────────

class TestIpToInt:
    def test_known_ip(self):
        # 192.168.1.1 → 3232235777
        assert ip_to_int("192.168.1.1") == 3232235777

    def test_zero_ip(self):
        assert ip_to_int("0.0.0.0") == 0

    def test_max_ip(self):
        assert ip_to_int("255.255.255.255") == 4294967295

    def test_invalid_ip(self):
        result = ip_to_int("not_an_ip")
        assert np.isnan(result)

    def test_empty_string(self):
        result = ip_to_int("")
        assert np.isnan(result)


# ─── merge_ip_country ───────────────────────────────────────────────────────

class TestMergeIpCountry:
    def _make_fraud_df(self):
        return pd.DataFrame({
            "user_id": [1, 2, 3],
            "ip_address": ["10.0.0.1", "192.168.1.5", "999.999.999.999"],
            "class": [0, 1, 0],
        })

    def _make_ip_country_df(self):
        return pd.DataFrame({
            "lower_bound_ip_address": [
                ip_to_int("10.0.0.0"),
                ip_to_int("192.168.1.0"),
            ],
            "upper_bound_ip_address": [
                ip_to_int("10.0.0.255"),
                ip_to_int("192.168.1.255"),
            ],
            "country": ["CountryA", "CountryB"],
        })

    def test_known_ip_matched(self):
        df = merge_ip_country(self._make_fraud_df(), self._make_ip_country_df())
        assert df.loc[df["user_id"] == 1, "country"].iloc[0] == "CountryA"
        assert df.loc[df["user_id"] == 2, "country"].iloc[0] == "CountryB"

    def test_unknown_ip_nan(self):
        df = merge_ip_country(self._make_fraud_df(), self._make_ip_country_df())
        assert pd.isna(df.loc[df["user_id"] == 3, "country"].iloc[0])

    def test_original_df_unchanged(self):
        fraud_df = self._make_fraud_df()
        _ = merge_ip_country(fraud_df, self._make_ip_country_df())
        assert "country" not in fraud_df.columns  # must not mutate input


# ─── engineer_features ──────────────────────────────────────────────────────

class TestEngineerFeatures:
    def _make_df(self):
        return pd.DataFrame({
            "user_id": [1, 1, 2],
            "signup_time": pd.to_datetime(["2023-01-01 08:00", "2023-01-01 08:00", "2023-01-02 10:00"]),
            "purchase_time": pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:30", "2023-01-02 11:00"]),
            "purchase_value": [50.0, 80.0, 20.0],
            "device_id": ["d1", "d1", "d2"],
            "source": ["SEO", "SEO", "Ads"],
            "browser": ["Chrome", "Chrome", "Firefox"],
            "sex": ["M", "M", "F"],
            "age": [25, 25, 30],
            "ip_address": ["1.2.3.4", "1.2.3.4", "5.6.7.8"],
            "country": ["US", "US", "UK"],
            "class": [0, 1, 0],
        })

    def test_time_since_signup_created(self):
        result = engineer_features(self._make_df())
        assert "time_since_signup_hours" in result.columns

    def test_time_since_signup_values(self):
        result = engineer_features(self._make_df())
        # user 1: 2h elapsed; user 2: 1h elapsed
        assert result["time_since_signup_hours"].iloc[0] == pytest.approx(2.0)
        assert result["time_since_signup_hours"].iloc[2] == pytest.approx(1.0)

    def test_hour_of_day(self):
        result = engineer_features(self._make_df())
        assert result["hour_of_day"].iloc[0] == 10

    def test_velocity_columns_exist(self):
        result = engineer_features(self._make_df())
        assert "transaction_count_1h" in result.columns
        assert "transaction_count_24h" in result.columns

    def test_no_negative_signup_hours(self):
        df = self._make_df()
        # Swap times so purchase is before signup
        df.loc[0, "purchase_time"] = pd.Timestamp("2022-12-31")
        result = engineer_features(df)
        assert (result["time_since_signup_hours"] >= 0).all()

    def test_raw_identifiers_dropped(self):
        result = engineer_features(self._make_df())
        for col in ["user_id", "device_id", "ip_address", "signup_time", "purchase_time"]:
            assert col not in result.columns


# ─── Resampling ─────────────────────────────────────────────────────────────

class TestResampling:
    def _make_imbalanced(self, n_majority=500, n_minority=50):
        np.random.seed(0)
        X = pd.DataFrame(np.random.randn(n_majority + n_minority, 5),
                         columns=list("abcde"))
        y = pd.Series([0] * n_majority + [1] * n_minority)
        return X, y

    def test_smote_balances_classes(self):
        X, y = self._make_imbalanced()
        X_res, y_res = apply_smote(X, y)
        counts = y_res.value_counts()
        assert counts[0] == counts[1]

    def test_smote_does_not_mutate_input(self):
        X, y = self._make_imbalanced()
        original_len = len(X)
        apply_smote(X, y)
        assert len(X) == original_len

    def test_undersampling_reduces_majority(self):
        X, y = self._make_imbalanced()
        X_res, y_res = apply_undersampling(X, y, sampling_strategy=0.5)
        counts = y_res.value_counts()
        # minority stays the same, majority is reduced
        assert counts[1] == 50
        assert counts[0] == 100  # 50 / 0.5 = 100
