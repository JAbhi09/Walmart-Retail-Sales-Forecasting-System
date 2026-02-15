"""
Train Walmart Sales Forecasting Model
Loads engineered features, trains LightGBM model, and evaluates performance
"""
import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import db_manager
from models.trainer import WalmartForecaster
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features_from_db():
    """Load engineered features from database"""
    logger.info("Loading engineered features from database...")
    
    engine = db_manager.connect()
    
    # Load all features, excluding rows with NULL in critical columns
    query = """
        SELECT *
        FROM engineered_features
        WHERE sales_lag_1 IS NOT NULL
        ORDER BY feature_date, store_id, dept_id
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"✓ Loaded {len(df):,} feature records")
    logger.info(f"  Date range: {df['feature_date'].min()} to {df['feature_date'].max()}")
    logger.info(f"  Stores: {df['store_id'].nunique()}")
    logger.info(f"  Departments: {df['dept_id'].nunique()}")
    
    return df


def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("WALMART SALES FORECASTING - MODEL TRAINING")
    logger.info("="*60)
    
    try:
        # Step 1: Load data
        logger.info("\n[1/4] Loading data...")
        df = load_features_from_db()
        
        # Step 2: Initialize forecaster
        logger.info("\n[2/4] Initializing forecaster...")
        forecaster = WalmartForecaster()
        
        # Step 3: Create train/validation split
        logger.info("\n[3/4] Creating train/validation split...")
        train_df, val_df = forecaster.create_train_val_split(df, val_weeks=8)
        
        # Step 4: Train model
        logger.info("\n[4/4] Training model...")
        metrics = forecaster.train(train_df, val_df)
        
        # Save model
        forecaster.save_model()
        
        logger.info("\n" + "="*60)
        logger.info("✓ MODEL TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Validation WMAE: {metrics['wmae']:.2f}")
        logger.info(f"Validation MAE:  {metrics['mae']:.2f}")
        logger.info(f"Validation RMSE: {metrics['rmse']:.2f}")
        logger.info(f"Validation MAPE: {metrics['mape']:.2f}%")
        
        # Check if we beat the baseline
        baseline_wmae = 821.0
        if metrics['wmae'] < baseline_wmae:
            improvement = ((baseline_wmae - metrics['wmae']) / baseline_wmae) * 100
            logger.info(f"\n🎉 Beat baseline! {improvement:.1f}% improvement over WMAE {baseline_wmae}")
        else:
            logger.info(f"\n⚠️  Did not beat baseline WMAE of {baseline_wmae}")
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
