"""
Pytest configuration and shared fixtures for testing.
"""
import os
import sys
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine with properly handled credentials."""
    url = URL.create(
        drivername="postgresql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),   # raw password — no manual encoding needed
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME"),
    )
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def sample_sales_data():
    """Generate sample sales data for testing."""
    dates = pd.date_range(start='2010-02-05', end='2012-10-26', freq='W-FRI')
    data = {
        'store_id': np.random.randint(1, 46, size=len(dates)),
        'dept_id': np.random.randint(1, 100, size=len(dates)),
        'sale_date': dates,
        'weekly_sales': np.random.uniform(1000, 50000, size=len(dates)),
        'is_holiday': np.random.choice([True, False], size=len(dates), p=[0.05, 0.95])
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_data():
    """Generate sample features data for testing."""
    dates = pd.date_range(start='2010-02-05', end='2012-10-26', freq='W-FRI')
    data = {
        'store_id': np.random.randint(1, 46, size=len(dates)),
        'feature_date': dates,
        'temperature': np.random.uniform(30, 100, size=len(dates)),
        'fuel_price': np.random.uniform(2.5, 4.5, size=len(dates)),
        'cpi': np.random.uniform(126, 228, size=len(dates)),
        'unemployment': np.random.uniform(3, 15, size=len(dates)),
        'is_holiday': np.random.choice([True, False], size=len(dates), p=[0.05, 0.95])
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_store_data():
    """Generate sample store data for testing."""
    data = {
        'store_id': list(range(1, 46)),
        'store_type': np.random.choice(['A', 'B', 'C'], size=45),
        'size_sqft': np.random.randint(30000, 220000, size=45)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_engineered_features():
    """Generate sample engineered features for testing."""
    dates = pd.date_range(start='2011-02-05', end='2012-10-26', freq='W-FRI')
    n_samples = len(dates)

    data = {
        'store_id': np.random.randint(1, 46, size=n_samples),
        'dept_id': np.random.randint(1, 100, size=n_samples),
        'feature_date': dates,
        'week_of_year': [d.isocalendar()[1] for d in dates],
        'month': [d.month for d in dates],
        'quarter': [d.quarter for d in dates],
        'sales_lag_1': np.random.uniform(1000, 50000, size=n_samples),
        'sales_lag_2': np.random.uniform(1000, 50000, size=n_samples),
        'rolling_mean_4': np.random.uniform(5000, 40000, size=n_samples),
        'rolling_std_4': np.random.uniform(500, 5000, size=n_samples),
        'temperature': np.random.uniform(30, 100, size=n_samples),
        'fuel_price': np.random.uniform(2.5, 4.5, size=n_samples),
        'cpi': np.random.uniform(126, 228, size=n_samples),
        'unemployment': np.random.uniform(3, 15, size=n_samples),
        'weekly_sales': np.random.uniform(1000, 50000, size=n_samples),
        'is_holiday': np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response for testing."""
    return {
        'text': 'This is a mock AI response for testing purposes.',
        'candidates': [
            {
                'content': {
                    'parts': [
                        {'text': 'Mock analysis: Sales are trending upward.'}
                    ]
                }
            }
        ]
    }