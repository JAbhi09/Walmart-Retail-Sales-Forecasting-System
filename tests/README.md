# Testing Guide

## Overview

This directory contains comprehensive tests for the Walmart Retail Sales Forecasting System.

## Test Structure

```
tests/
├── __init__.py                    # Package initialization
├── conftest.py                    # Pytest configuration and fixtures
├── test_database.py               # Database connectivity and operations
├── test_feature_engineering.py    # Feature engineering pipeline
├── test_model.py                  # ML model training and prediction
├── test_agents.py                 # AI agents functionality
├── test_integration.py            # End-to-end integration tests
└── test_validation.py             # Performance and validation tests
```

## Prerequisites

Install testing dependencies:
```bash
pip install pytest pytest-cov pytest-mock
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_database.py
pytest tests/test_model.py
```

### Run with Coverage Report
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Specific Test Class or Function
```bash
pytest tests/test_database.py::TestDatabaseConnection
pytest tests/test_model.py::TestMetrics::test_wmae_calculation
```

## Test Categories

### 1. Unit Tests
- **test_database.py**: Database connections, CRUD operations, constraints
- **test_feature_engineering.py**: Feature creation, validation, pipeline
- **test_model.py**: Model training, metrics, predictions, persistence
- **test_agents.py**: Agent initialization, calculations, AI integration

### 2. Integration Tests
- **test_integration.py**: End-to-end pipelines, data consistency, error handling

### 3. Performance Tests
- **test_validation.py**: Forecast accuracy, model robustness, scalability

## Test Fixtures

Common fixtures available in `conftest.py`:
- `test_db_engine`: Database connection for testing
- `sample_sales_data`: Sample sales DataFrame
- `sample_features_data`: Sample features DataFrame
- `sample_store_data`: Sample store DataFrame
- `sample_engineered_features`: Sample engineered features
- `mock_gemini_response`: Mock AI response

## Environment Setup

Ensure `.env` file is configured with test database credentials:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=walmart_retail
DB_USER=postgres
DB_PASSWORD=your_password
```

## Test Coverage Goals

- **Database**: 80%+ coverage
- **Feature Engineering**: 85%+ coverage
- **ML Models**: 90%+ coverage
- **AI Agents**: 75%+ coverage (excluding external API calls)
- **Integration**: 70%+ coverage

## Skipped Tests

Some tests are skipped by default:
- Tests requiring Streamlit server
- Tests requiring Gemini API key (to avoid API costs)

To run skipped tests:
```bash
pytest tests/ --run-skipped
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov
```

## Troubleshooting

### Database Connection Errors
- Ensure PostgreSQL is running
- Verify credentials in `.env`
- Check database exists: `walmart_retail`

### Import Errors
- Ensure project root is in PYTHONPATH
- Run from project root directory

### Fixture Errors
- Check `conftest.py` is present
- Verify pytest version: `pytest --version`

## Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Use fixtures for setup/teardown
3. **Mocking**: Mock external APIs (Gemini) to avoid costs
4. **Performance**: Keep tests fast (< 1 second each)
5. **Assertions**: Use descriptive assertion messages

## Adding New Tests

1. Create test file: `test_<module>.py`
2. Import required fixtures from `conftest.py`
3. Organize tests into classes by functionality
4. Use descriptive test names: `test_<what>_<expected_behavior>`
5. Add docstrings explaining test purpose

Example:
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_feature_works_correctly(self, sample_data):
        """Test that new feature produces expected output."""
        result = new_feature(sample_data)
        assert result is not None
```
