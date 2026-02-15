"""
Training Script for Walmart Sales Forecaster
Loads data, trains LightGBM model, saves to disk and MLflow
"""
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from models.trainer import WalmartForecaster


def get_db_engine():
    """Create database engine with URL-encoded password"""
    password = quote_plus(os.getenv('DB_PASSWORD'))
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{password}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def load_training_data(engine):
    """Load engineered features from database"""
    logger.info("Loading training data from database...")

    query = """
        SELECT *
        FROM engineered_features
        ORDER BY store_id, dept_id, feature_date
    """

    df = pd.read_sql(query, engine)
    logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"   Date range: {df['feature_date'].min()} to {df['feature_date'].max()}")
    logger.info(f"   Stores: {df['store_id'].nunique()}, Departments: {df['dept_id'].nunique()}")

    return df


def clean_data(df):
    """Remove rows with null targets and key lag features"""
    logger.info("Cleaning data...")

    initial = len(df)

    # Drop rows where target is null
    df = df.dropna(subset=['weekly_sales'])

    # Drop rows where critical lag features are null
    # (these are the first few weeks per store-dept with no history)
    lag_cols = [c for c in df.columns if 'lag' in c or 'rolling' in c]
    df = df.dropna(subset=lag_cols, how='all')

    # Fill remaining NaN in numeric columns with 0
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Convert boolean columns to int for LightGBM
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    logger.info(f"✅ Cleaned: {initial:,} → {len(df):,} rows ({initial - len(df):,} removed)")

    return df


def main():
    logger.info("=" * 60)
    logger.info("WALMART SALES FORECASTER - TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Connect to database
    logger.info("\n[1/5] Connecting to database...")
    engine = get_db_engine()

    # 2. Load data
    logger.info("\n[2/5] Loading training data...")
    df = load_training_data(engine)

    # 3. Clean data
    logger.info("\n[3/5] Cleaning data...")
    df = clean_data(df)

    # 4. Initialize forecaster and train
    logger.info("\n[4/5] Training model...")
    forecaster = WalmartForecaster(config_path='config/config.yaml')

    # Create time-based train/val split (last 8 weeks for validation)
    train_df, val_df = forecaster.create_train_val_split(df, val_weeks=8)

    # Train with validation
    metrics = forecaster.train(train_df, val_df)

    # 5. Save model locally
    logger.info("\n[5/5] Saving model...")
    forecaster.save_model('models/saved/walmart_forecaster.txt')

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"WMAE:  {metrics['wmae']:.2f}  (target: < 700, baseline: 821)")
    logger.info(f"MAE:   {metrics['mae']:.2f}")
    logger.info(f"RMSE:  {metrics['rmse']:.2f}")
    logger.info(f"MAPE:  {metrics['mape']:.2f}%")
    logger.info(f"\nModel saved to: models/saved/walmart_forecaster.txt")
    logger.info(f"MLflow UI: {os.getenv('MLFLOW_TRACKING_URI')}")
    logger.info(f"\nNext step: python models/generate_forecasts.py")


if __name__ == "__main__":
    main()