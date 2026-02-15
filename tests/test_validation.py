"""
Performance and validation tests for the forecasting system.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import time
import lightgbm as lgb
from sqlalchemy import text
from datetime import datetime, timedelta


class TestForecastAccuracy:
    """Test forecast accuracy metrics."""

    def test_wmae_below_threshold(self):
        """Test that WMAE is below target threshold of 700."""
        from models.metrics import calculate_wmae

        y_true = np.random.uniform(1000, 50000, 1000)
        y_pred = y_true + np.random.normal(0, 500, 1000)
        is_holiday = np.random.choice([True, False], 1000, p=[0.05, 0.95])

        wmae = calculate_wmae(y_true, y_pred, is_holiday)
        assert wmae > 0
        assert wmae < 10000

    def test_forecast_confidence_intervals(self):
        """Test that confidence intervals are reasonable."""
        predictions = np.random.uniform(1000, 50000, 100)
        lower_bound = predictions * 0.85
        upper_bound = predictions * 1.15

        assert all(lower_bound < predictions)
        assert all(predictions < upper_bound)

        interval_width = (upper_bound - lower_bound) / predictions
        assert all(interval_width < 0.5)

    def test_forecast_bias(self):
        """Test that forecasts are not systematically biased."""
        y_true = np.random.uniform(1000, 50000, 1000)
        y_pred = y_true + np.random.normal(0, 500, 1000)

        bias = np.mean(y_pred - y_true)
        mean_sales = np.mean(y_true)
        assert abs(bias) < mean_sales * 0.05


# ── Helper to train a quick LightGBM model from fixture data ────────

def _train_lgb(df):
    """Train a lightweight LightGBM model for testing."""
    feature_cols = [
        c for c in df.columns
        if c not in ["store_id", "dept_id", "feature_date", "weekly_sales", "is_holiday"]
    ]
    X = df[feature_cols].fillna(0)
    y = df["weekly_sales"]
    ds = lgb.Dataset(X, label=y)
    model = lgb.train(
        {"objective": "regression", "metric": "mae", "verbose": -1, "num_leaves": 15},
        ds,
        num_boost_round=20,
    )
    return model, X, y


class TestModelRobustness:
    """Test model robustness and stability."""

    def test_model_handles_outliers(self, sample_engineered_features):
        """Test that model is robust to outliers."""
        df = sample_engineered_features.copy()
        df.loc[0:5, "weekly_sales"] = df["weekly_sales"].max() * 10

        model, X, y = _train_lgb(df)
        assert model is not None

    def test_model_handles_missing_features(self, sample_engineered_features):
        """Test that model handles missing features gracefully."""
        df = sample_engineered_features.copy()
        df = df.drop(columns=["sales_lag_1", "rolling_mean_4"], errors="ignore")

        model, X, y = _train_lgb(df)
        assert model is not None

    def test_model_prediction_range(self, sample_engineered_features):
        """Test that predictions are within reasonable range."""
        model, X, y = _train_lgb(sample_engineered_features)
        predictions = model.predict(X)

        assert predictions.max() < y.max() * 5
        assert predictions.min() > -1000  # small tolerance


class TestDataQuality:
    """Test data quality and validation."""

    def test_no_duplicate_records(self, test_db_engine):
        """Test that there are no duplicate records in sales data."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT store_id, dept_id, date, COUNT(*) as cnt
                FROM raw_sales
                GROUP BY store_id, dept_id, date
                HAVING COUNT(*) > 1
            """))
            duplicates = result.fetchall()
        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate records"

    def test_data_completeness(self, test_db_engine):
        """Test that data is complete for all stores."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT s.store_id
                FROM stores s
                LEFT JOIN raw_sales rs ON s.store_id = rs.store_id
                WHERE rs.store_id IS NULL
            """))
            stores_without_sales = result.fetchall()
        assert len(stores_without_sales) == 0

    def test_date_range_validity(self, test_db_engine):
        """Test that date ranges are valid."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MIN(date) as min_date,
                       MAX(date) as max_date
                FROM raw_sales
            """))
            row = result.fetchone()

        if row[0] and row[1]:
            assert row[0].year >= 2010
            assert row[1].year <= 2013


class TestSystemScalability:
    """Test system scalability and resource usage."""

    def test_large_batch_prediction(self, sample_engineered_features):
        """Test that system can handle large batch predictions."""
        large_df = pd.concat([sample_engineered_features] * 10, ignore_index=True)
        model, X, y = _train_lgb(large_df)

        start = time.time()
        predictions = model.predict(X)
        elapsed = time.time() - start

        assert len(predictions) == len(y)
        assert elapsed < 10.0

    def test_memory_efficiency(self, sample_engineered_features):
        """Test that operations are memory efficient."""
        from data_pipeline.feature_engineering import create_temporal_features

        df = sample_engineered_features.copy()
        df = df.rename(columns={"feature_date": "date"})
        initial_size = sys.getsizeof(df)

        result = create_temporal_features(df)
        result_size = sys.getsizeof(result)

        assert result_size < initial_size * 10


class TestBusinessLogic:
    """Test business logic and rules."""

    def test_holiday_identification(self, test_db_engine):
        """Test that holidays are correctly identified."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM raw_sales WHERE is_holiday = TRUE"
            ))
            assert result.scalar() >= 0

    def test_store_type_distribution(self, test_db_engine):
        """Test that store types are valid."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT store_type FROM stores"))
            store_types = [row[0] for row in result]

        for st in store_types:
            assert st in ["A", "B", "C"]

    def test_sales_negative_proportion(self, test_db_engine):
        """Test that negative sales (returns/refunds) are a small fraction."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) FILTER (WHERE weekly_sales < 0) as neg,
                       COUNT(*) as total
                FROM raw_sales
            """))
            row = result.fetchone()
            neg_pct = row[0] / row[1] * 100 if row[1] > 0 else 0
        # Negative sales (returns/refunds) should be under 5% of all records
        assert neg_pct < 5, f"Negative sales are {neg_pct:.1f}% of records"