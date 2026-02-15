"""
Unit tests for feature engineering pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_pipeline.feature_engineering import (
    create_temporal_features,
    create_lag_features,
    create_rolling_features,
    create_economic_features,
    create_markdown_features,
    create_store_features,
    engineer_features,
)


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def sample_sales_data():
    """Generate minimal sales DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-06", periods=52, freq="W-FRI")
    rows = []
    for store in [1, 2]:
        for dept in [1, 2]:
            for d in dates:
                rows.append({
                    "store_id": store,
                    "dept_id": dept,
                    "date": d,
                    "weekly_sales": np.random.uniform(5000, 50000),
                    "is_holiday": np.random.choice([True, False], p=[0.1, 0.9]),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_features_data():
    """Generate minimal features DataFrame (one row per store-date)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-06", periods=52, freq="W-FRI")
    rows = []
    for store in [1, 2]:
        for d in dates:
            rows.append({
                "store_id": store,
                "date": d,
                "temperature": np.random.uniform(30, 100),
                "fuel_price": np.random.uniform(2.5, 4.5),
                "cpi": np.random.uniform(200, 250),
                "unemployment": np.random.uniform(4, 10),
                "is_holiday": np.random.choice([True, False], p=[0.1, 0.9]),
                "markdown1": np.random.choice([0, np.nan, 1000, 2500]),
                "markdown2": np.random.choice([0, np.nan, 500, 1500]),
                "markdown3": np.random.choice([0, np.nan, 300]),
                "markdown4": np.random.choice([0, np.nan, 800]),
                "markdown5": np.random.choice([0, np.nan, 400]),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_stores_data():
    """Generate minimal stores DataFrame."""
    return pd.DataFrame({
        "store_id": [1, 2],
        "store_type": ["A", "B"],
        "size": [150000, 100000],
    })


# ── Temporal Features ───────────────────────────────────────

class TestTemporalFeatures:
    """Test create_temporal_features."""

    def test_columns_created(self, sample_sales_data):
        result = create_temporal_features(sample_sales_data)
        for col in ["week_of_year", "month", "quarter", "is_month_start", "is_month_end"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_value_ranges(self, sample_sales_data):
        result = create_temporal_features(sample_sales_data)
        assert result["month"].between(1, 12).all()
        assert result["quarter"].between(1, 4).all()

    def test_does_not_mutate_input(self, sample_sales_data):
        original_cols = list(sample_sales_data.columns)
        create_temporal_features(sample_sales_data)
        assert list(sample_sales_data.columns) == original_cols


# ── Lag Features ────────────────────────────────────────────

class TestLagFeatures:
    """Test create_lag_features."""

    def test_lag_columns_created(self, sample_sales_data):
        result = create_lag_features(sample_sales_data, lags=[1, 2, 4])
        for lag in [1, 2, 4]:
            assert f"sales_lag_{lag}" in result.columns

    def test_first_rows_have_nans(self, sample_sales_data):
        result = create_lag_features(sample_sales_data, lags=[1])
        # First row of each group should be NaN (use nth to avoid skipping NaN)
        first_rows = result.groupby(["store_id", "dept_id"]).nth(0)
        assert first_rows["sales_lag_1"].isna().all()

    def test_does_not_mutate_input(self, sample_sales_data):
        original_cols = list(sample_sales_data.columns)
        create_lag_features(sample_sales_data, lags=[1])
        assert list(sample_sales_data.columns) == original_cols


# ── Rolling Features ────────────────────────────────────────

class TestRollingFeatures:
    """Test create_rolling_features."""

    def test_rolling_columns_created(self, sample_sales_data):
        result = create_rolling_features(sample_sales_data, windows=[4, 13])
        assert "rolling_mean_4" in result.columns
        assert "rolling_std_4" in result.columns
        assert "rolling_mean_13" in result.columns

    def test_4week_window_has_min_max(self, sample_sales_data):
        result = create_rolling_features(sample_sales_data, windows=[4])
        assert "rolling_min_4" in result.columns
        assert "rolling_max_4" in result.columns

    def test_rolling_mean_is_reasonable(self, sample_sales_data):
        result = create_rolling_features(sample_sales_data, windows=[4])
        assert result["rolling_mean_4"].notna().sum() > 0
        # Rolling mean should be within the range of weekly_sales
        assert result["rolling_mean_4"].min() >= sample_sales_data["weekly_sales"].min()


# ── Economic Features ───────────────────────────────────────

class TestEconomicFeatures:
    """Test create_economic_features."""

    def test_columns_created(self, sample_features_data):
        result = create_economic_features(sample_features_data)
        for col in ["temperature_deviation", "fuel_price_change", "cpi_change", "unemployment_change"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_temperature_deviation_centered(self, sample_features_data):
        result = create_economic_features(sample_features_data)
        # Mean deviation per store should be ~0
        mean_dev = result.groupby("store_id")["temperature_deviation"].mean()
        assert (mean_dev.abs() < 1e-6).all()


# ── Markdown Features ───────────────────────────────────────

class TestMarkdownFeatures:
    """Test create_markdown_features."""

    def test_columns_created(self, sample_features_data):
        result = create_markdown_features(sample_features_data)
        for col in ["total_markdown", "has_markdown", "markdown_count"]:
            assert col in result.columns

    def test_total_markdown_non_negative(self, sample_features_data):
        result = create_markdown_features(sample_features_data)
        assert (result["total_markdown"] >= 0).all()

    def test_has_markdown_is_binary(self, sample_features_data):
        result = create_markdown_features(sample_features_data)
        assert set(result["has_markdown"].unique()).issubset({0, 1})

    def test_nan_markdowns_filled(self, sample_features_data):
        result = create_markdown_features(sample_features_data)
        md_cols = [c for c in result.columns if c.startswith("markdown") and c[-1].isdigit()]
        for col in md_cols:
            assert result[col].isna().sum() == 0


# ── Store Features ──────────────────────────────────────────

class TestStoreFeatures:
    """Test create_store_features."""

    def test_columns_created(self, sample_sales_data, sample_stores_data):
        result = create_store_features(sample_sales_data, sample_stores_data)
        for col in ["store_type_a", "store_type_b", "store_type_c", "size_normalized"]:
            assert col in result.columns

    def test_store_type_one_hot(self, sample_sales_data, sample_stores_data):
        result = create_store_features(sample_sales_data, sample_stores_data)
        # Each row should have exactly one store type = 1
        type_sum = result[["store_type_a", "store_type_b", "store_type_c"]].sum(axis=1)
        assert (type_sum == 1).all()

    def test_original_columns_dropped(self, sample_sales_data, sample_stores_data):
        result = create_store_features(sample_sales_data, sample_stores_data)
        assert "store_type" not in result.columns
        assert "size" not in result.columns


# ── Feature Validation ──────────────────────────────────────

class TestFeatureValidation:
    """Test feature quality checks."""

    def test_no_infinite_values(self, sample_sales_data):
        result = create_rolling_features(sample_sales_data, windows=[4])
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(result[col]).any(), f"{col} contains infinite values"

    def test_lag_features_correct_dtype(self, sample_sales_data):
        result = create_lag_features(sample_sales_data, lags=[1, 2])
        for col in ["sales_lag_1", "sales_lag_2"]:
            assert pd.api.types.is_numeric_dtype(result[col])


# ── Full Pipeline ───────────────────────────────────────────

class TestFullPipeline:
    """Test engineer_features end-to-end."""

    def test_pipeline_runs(self, sample_sales_data, sample_features_data, sample_stores_data):
        result = engineer_features(sample_sales_data, sample_features_data, sample_stores_data)
        assert len(result) > 0
        # Should have more columns than the raw sales data
        assert result.shape[1] > sample_sales_data.shape[1]

    def test_pipeline_preserves_rows(self, sample_sales_data, sample_features_data, sample_stores_data):
        result = engineer_features(sample_sales_data, sample_features_data, sample_stores_data)
        # Should not lose rows (left join)
        assert len(result) == len(sample_sales_data)

    def test_pipeline_renames_date(self, sample_sales_data, sample_features_data, sample_stores_data):
        result = engineer_features(sample_sales_data, sample_features_data, sample_stores_data)
        assert "feature_date" in result.columns
        assert "date" not in result.columns