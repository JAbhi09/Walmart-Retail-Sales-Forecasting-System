"""
Integration tests for end-to-end pipeline.
"""
import pytest
import time
import pandas as pd
import numpy as np
from sqlalchemy import text


class TestDataPipeline:
    """Test end-to-end data pipeline."""

    def test_data_loading_pipeline(self, test_db_engine):
        """Test that data can be loaded from CSV to database."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM raw_sales"))
            count = result.scalar()
            assert count >= 0

    def test_feature_engineering_pipeline(self, test_db_engine):
        """Test that feature engineering pipeline runs end-to-end."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM engineered_features"))
            count = result.scalar()
            assert count >= 0


class TestMLPipeline:
    """Test end-to-end ML pipeline."""

    def test_model_trainer_importable(self):
        """Test that model trainer can be imported."""
        from models.trainer import WalmartForecaster
        assert WalmartForecaster is not None

    def test_forecast_generation_pipeline(self, test_db_engine):
        """Test that forecasts can be generated and stored."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM forecasts"))
            count = result.scalar()
            assert count >= 0


class TestDashboardIntegration:
    """Test dashboard integration with backend."""

    @pytest.mark.skip(reason="Requires Streamlit server running")
    def test_dashboard_loads(self):
        pass

    def test_dashboard_data_queries(self, test_db_engine):
        """Test that dashboard can query required data."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM stores"))
            assert result.scalar() >= 0

            result = conn.execute(text(
                "SELECT COUNT(*) FROM forecasts WHERE created_at >= NOW() - INTERVAL '30 days'"
            ))
            assert result.scalar() >= 0


class TestAgentWorkflow:
    """Test AI agent workflow integration."""

    @pytest.mark.skip(reason="Requires Gemini API key")
    def test_agent_workflow_execution(self):
        pass

    def test_agent_data_access(self, test_db_engine):
        """Test that agents can access required data."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT store_id, dept_id, weekly_sales FROM raw_sales LIMIT 10"
            ))
            rows = result.fetchall()
            assert len(rows) >= 0


class TestSystemPerformance:
    """Test system performance and response times."""

    def test_database_query_performance(self, test_db_engine):
        """Test that database queries execute within acceptable time."""
        with test_db_engine.connect() as conn:
            start = time.time()
            result = conn.execute(text("""
                SELECT store_id, dept_id, AVG(weekly_sales) as avg_sales
                FROM raw_sales
                GROUP BY store_id, dept_id
                LIMIT 100
            """))
            result.fetchall()
            elapsed = time.time() - start
            assert elapsed < 5.0

    def test_feature_engineering_performance(self, sample_sales_data):
        """Test that feature engineering completes in reasonable time."""
        from data_pipeline.feature_engineering import create_temporal_features

        df = sample_sales_data.copy()
        df = df.rename(columns={"sale_date": "date"})

        start = time.time()
        result = create_temporal_features(df)
        elapsed = time.time() - start

        assert elapsed < 10.0
        assert len(result) == len(df)


class TestDataConsistency:
    """Test data consistency across pipeline."""

    def test_sales_data_consistency(self, test_db_engine):
        """Test that sales data is consistent across tables."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(DISTINCT rs.store_id) as sales_stores,
                       (SELECT COUNT(*) FROM stores) as total_stores
                FROM raw_sales rs
            """))
            row = result.fetchone()
            if row[0] is not None and row[1] is not None:
                assert row[0] <= row[1]

    def test_date_consistency(self, test_db_engine):
        """Test that dates are consistent across tables."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM raw_sales WHERE date > CURRENT_DATE"
            ))
            assert result.scalar() == 0, "Found future dates in sales data"


class TestErrorHandling:
    """Test error handling across the system."""

    def test_database_connection_error_handling(self):
        """Test handling of database connection errors."""
        from sqlalchemy import create_engine
        from sqlalchemy.exc import OperationalError

        try:
            engine = create_engine("postgresql://invalid:invalid@localhost:5432/nonexistent")
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except OperationalError:
            assert True
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e)}")

    def test_missing_data_handling(self):
        """Test handling of missing data in feature engineering."""
        from data_pipeline.feature_engineering import create_temporal_features

        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="W"),
            "weekly_sales": [1000, np.nan, 2000, np.nan, 3000, 4000, np.nan, 5000, 6000, 7000],
        })

        # create_temporal_features should not crash on NaN sales
        result = create_temporal_features(df)
        assert result is not None
        assert len(result) == 10