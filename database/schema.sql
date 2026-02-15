-- Walmart Retail Forecasting Database Schema

-- Drop existing tables if they exist
DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS engineered_features CASCADE;
DROP TABLE IF EXISTS model_metadata CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS raw_sales CASCADE;
DROP TABLE IF EXISTS stores CASCADE;

-- Stores table
CREATE TABLE stores (
    store_id INTEGER PRIMARY KEY,
    store_type VARCHAR(1) NOT NULL CHECK (store_type IN ('A', 'B', 'C')),
    size INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw sales data
CREATE TABLE raw_sales (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(store_id),
    dept_id INTEGER NOT NULL,
    date DATE NOT NULL,
    weekly_sales DECIMAL(12, 2) NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, dept_id, date)
);

-- Features (economic indicators, markdowns)
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(store_id),
    date DATE NOT NULL,
    temperature DECIMAL(5, 2),
    fuel_price DECIMAL(5, 3),
    markdown1 DECIMAL(12, 2),
    markdown2 DECIMAL(12, 2),
    markdown3 DECIMAL(12, 2),
    markdown4 DECIMAL(12, 2),
    markdown5 DECIMAL(12, 2),
    cpi DECIMAL(12, 7),
    unemployment DECIMAL(5, 3),
    is_holiday BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, date)
);

-- Engineered features (ML-ready)
CREATE TABLE engineered_features (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL,
    dept_id INTEGER NOT NULL,
    feature_date DATE NOT NULL,
    
    -- Target
    weekly_sales DECIMAL(12, 2),
    
    -- Temporal features
    week_of_year INTEGER,
    month INTEGER,
    quarter INTEGER,
    is_month_start BOOLEAN,
    is_month_end BOOLEAN,
    is_holiday BOOLEAN,
    
    -- Lag features
    sales_lag_1 DECIMAL(12, 2),
    sales_lag_2 DECIMAL(12, 2),
    sales_lag_4 DECIMAL(12, 2),
    sales_lag_8 DECIMAL(12, 2),
    sales_lag_52 DECIMAL(12, 2),
    
    -- Rolling statistics
    rolling_mean_4 DECIMAL(12, 2),
    rolling_mean_13 DECIMAL(12, 2),
    rolling_mean_52 DECIMAL(12, 2),
    rolling_std_4 DECIMAL(12, 2),
    rolling_std_13 DECIMAL(12, 2),
    rolling_min_4 DECIMAL(12, 2),
    rolling_max_4 DECIMAL(12, 2),
    
    -- Economic indicators
    temperature DECIMAL(5, 2),
    temperature_deviation DECIMAL(5, 2),
    fuel_price DECIMAL(5, 3),
    fuel_price_change DECIMAL(5, 3),
    cpi DECIMAL(12, 7),
    cpi_change DECIMAL(12, 7),
    unemployment DECIMAL(5, 3),
    unemployment_change DECIMAL(5, 3),
    
    -- Markdown features
    total_markdown DECIMAL(12, 2),
    has_markdown BOOLEAN,
    markdown_count INTEGER,
    
    -- Store features
    store_type_A BOOLEAN,
    store_type_B BOOLEAN,
    store_type_C BOOLEAN,
    size_normalized DECIMAL(10, 6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, dept_id, feature_date)
);

-- Forecasts
CREATE TABLE forecasts (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL,
    dept_id INTEGER NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_sales DECIMAL(12, 2) NOT NULL,
    prediction_lower DECIMAL(12, 2),
    prediction_upper DECIMAL(12, 2),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    confidence_score DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, dept_id, forecast_date, model_name, created_at)
);

-- Model metadata
CREATE TABLE model_metadata (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    wmae DECIMAL(10, 2),
    mae DECIMAL(10, 2),
    rmse DECIMAL(10, 2),
    training_date TIMESTAMP NOT NULL,
    parameters JSONB,
    feature_importance JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_raw_sales_store_dept_date ON raw_sales(store_id, dept_id, date);
CREATE INDEX idx_raw_sales_date ON raw_sales(date);
CREATE INDEX idx_features_store_date ON features(store_id, date);
CREATE INDEX idx_engineered_features_store_dept_date ON engineered_features(store_id, dept_id, feature_date);
CREATE INDEX idx_forecasts_store_dept_date ON forecasts(store_id, dept_id, forecast_date);
CREATE INDEX idx_forecasts_date ON forecasts(forecast_date);
