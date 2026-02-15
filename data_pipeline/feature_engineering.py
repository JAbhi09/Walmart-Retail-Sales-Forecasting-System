"""
Feature Engineering Module
Transforms raw sales and features data into ML-ready features
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_temporal_features(df):
    """
    Create temporal features from date column
    
    Args:
        df: DataFrame with 'date' column
    
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    
    logger.info("✓ Created temporal features")
    return df


def create_lag_features(df, lags=[1, 2, 4, 8, 52], target_col='weekly_sales'):
    """
    Create lag features for time series
    
    Args:
        df: DataFrame sorted by store_id, dept_id, date
        lags: List of lag periods (in weeks)
        target_col: Column to create lags for
    
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    # Sort to ensure proper lag calculation
    df = df.sort_values(['store_id', 'dept_id', 'date'])
    
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store_id', 'dept_id'])[target_col].shift(lag)
    
    logger.info(f"✓ Created {len(lags)} lag features")
    return df


def create_rolling_features(df, windows=[4, 13, 52], target_col='weekly_sales'):
    """
    Create rolling window statistics
    
    Args:
        df: DataFrame sorted by store_id, dept_id, date
        windows: List of window sizes (in weeks)
        target_col: Column to calculate rolling stats for
    
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    # Sort to ensure proper rolling calculation
    df = df.sort_values(['store_id', 'dept_id', 'date'])
    
    for window in windows:
        # Rolling mean
        df[f'rolling_mean_{window}'] = df.groupby(['store_id', 'dept_id'])[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        df[f'rolling_std_{window}'] = df.groupby(['store_id', 'dept_id'])[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Additional stats for 4-week window
        if window == 4:
            df[f'rolling_min_{window}'] = df.groupby(['store_id', 'dept_id'])[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f'rolling_max_{window}'] = df.groupby(['store_id', 'dept_id'])[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
    
    logger.info(f"✓ Created rolling features for {len(windows)} windows")
    return df


def create_economic_features(df):
    """
    Create features from economic indicators
    
    Args:
        df: DataFrame with temperature, fuel_price, cpi, unemployment
    
    Returns:
        DataFrame with economic features
    """
    df = df.copy()
    
    # Sort for proper shift calculations
    df = df.sort_values(['store_id', 'date'])
    
    # Temperature deviation from mean
    df['temperature_deviation'] = df.groupby('store_id')['temperature'].transform(
        lambda x: x - x.mean()
    )
    
    # Fuel price change
    df['fuel_price_change'] = df.groupby('store_id')['fuel_price'].transform(
        lambda x: x.diff()
    )
    
    # CPI change
    df['cpi_change'] = df.groupby('store_id')['cpi'].transform(
        lambda x: x.diff()
    )
    
    # Unemployment change
    df['unemployment_change'] = df.groupby('store_id')['unemployment'].transform(
        lambda x: x.diff()
    )
    
    logger.info("✓ Created economic indicator features")
    return df


def create_markdown_features(df):
    """
    Create features from markdown columns
    
    Args:
        df: DataFrame with markdown1-5 columns
    
    Returns:
        DataFrame with markdown features
    """
    df = df.copy()
    
    markdown_cols = ['markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5']
    
    # Fill NaN with 0 for markdowns
    for col in markdown_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Total markdown
    df['total_markdown'] = df[markdown_cols].sum(axis=1)
    
    # Has markdown flag
    df['has_markdown'] = (df['total_markdown'] > 0).astype(int)
    
    # Count of active markdowns
    df['markdown_count'] = (df[markdown_cols] > 0).sum(axis=1)
    
    logger.info("✓ Created markdown features")
    return df


def create_store_features(df, stores_df):
    """
    Create features from store metadata
    
    Args:
        df: DataFrame with store_id
        stores_df: DataFrame with store metadata (type, size)
    
    Returns:
        DataFrame with store features
    """
    df = df.copy()
    
    # Merge store info
    df = df.merge(stores_df[['store_id', 'store_type', 'size']], on='store_id', how='left')
    
    # One-hot encode store type
    df['store_type_a'] = (df['store_type'] == 'A').astype(int)
    df['store_type_b'] = (df['store_type'] == 'B').astype(int)
    df['store_type_c'] = (df['store_type'] == 'C').astype(int)
    
    # Normalize store size
    df['size_normalized'] = (df['size'] - df['size'].mean()) / df['size'].std()
    
    # Drop original columns
    df = df.drop(['store_type', 'size'], axis=1)
    
    logger.info("✓ Created store features")
    return df


def engineer_features(sales_df, features_df, stores_df):
    """
    Complete feature engineering pipeline
    
    Args:
        sales_df: Raw sales data
        features_df: Economic indicators and markdowns
        stores_df: Store metadata
    
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Merge sales with features
    df = sales_df.merge(
        features_df,
        left_on=['store_id', 'date'],
        right_on=['store_id', 'date'],
        how='left',
        suffixes=('', '_feat')
    )
    
    # Use is_holiday from sales (more granular - by department)
    if 'is_holiday_feat' in df.columns:
        df = df.drop('is_holiday_feat', axis=1)
    
    logger.info(f"Merged data shape: {df.shape}")
    
    # 1. Temporal features
    df = create_temporal_features(df)
    
    # 2. Lag features
    df = create_lag_features(df, lags=[1, 2, 4, 8, 52])
    
    # 3. Rolling features
    df = create_rolling_features(df, windows=[4, 13, 52])
    
    # 4. Economic features
    df = create_economic_features(df)
    
    # 5. Markdown features
    df = create_markdown_features(df)
    
    # 6. Store features
    df = create_store_features(df, stores_df)
    
    # Rename date to feature_date for clarity
    df = df.rename(columns={'date': 'feature_date'})
    
    logger.info(f"✓ Feature engineering complete. Final shape: {df.shape}")
    logger.info(f"  Features created: {df.shape[1]} columns")
    logger.info(f"  Records: {len(df):,}")
    
    return df
