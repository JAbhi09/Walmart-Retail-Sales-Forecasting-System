"""
Run Feature Engineering Pipeline
Reads raw data from database, engineers features, and writes back to database
"""
import os
import sys
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import db_manager
from data_pipeline.feature_engineering import engineer_features
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_db():
    """Load raw data from database"""
    logger.info("Loading data from database...")
    
    engine = db_manager.connect()
    
    # Load sales
    sales_df = pd.read_sql("""
        SELECT store_id, dept_id, date, weekly_sales, is_holiday
        FROM raw_sales
        ORDER BY store_id, dept_id, date
    """, engine)
    logger.info(f"  Loaded {len(sales_df):,} sales records")
    
    # Load features
    features_df = pd.read_sql("""
        SELECT store_id, date, temperature, fuel_price,
               markdown1, markdown2, markdown3, markdown4, markdown5,
               cpi, unemployment, is_holiday
        FROM features
        ORDER BY store_id, date
    """, engine)
    logger.info(f"  Loaded {len(features_df):,} feature records")
    
    # Load stores
    stores_df = pd.read_sql("""
        SELECT store_id, store_type, size
        FROM stores
        ORDER BY store_id
    """, engine)
    logger.info(f"  Loaded {len(stores_df)} stores")
    
    return sales_df, features_df, stores_df


def write_features_to_db(features_df):
    """Write engineered features to database"""
    logger.info("Writing engineered features to database...")
    
    # Select columns in the correct order for the database schema
    columns = [
        'store_id', 'dept_id', 'feature_date', 'weekly_sales',
        'week_of_year', 'month', 'quarter', 'is_month_start', 'is_month_end', 'is_holiday',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_8', 'sales_lag_52',
        'rolling_mean_4', 'rolling_mean_13', 'rolling_mean_52',
        'rolling_std_4', 'rolling_std_13', 'rolling_min_4', 'rolling_max_4',
        'temperature', 'temperature_deviation', 'fuel_price', 'fuel_price_change',
        'cpi', 'cpi_change', 'unemployment', 'unemployment_change',
        'total_markdown', 'has_markdown', 'markdown_count',
        'store_type_a', 'store_type_b', 'store_type_c', 'size_normalized'
    ]
    
    # Ensure all columns exist
    for col in columns:
        if col not in features_df.columns:
            logger.warning(f"  Missing column: {col}, filling with NULL")
            features_df[col] = None
    
    # Select and reorder columns
    features_df = features_df[columns]
    
    # Ensure boolean columns are proper boolean type for PostgreSQL
    bool_cols = ['is_month_start', 'is_month_end', 'is_holiday', 
                 'has_markdown', 'store_type_a', 'store_type_b', 'store_type_c']
    for col in bool_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(bool)
    
    # Write to database in chunks
    engine = db_manager.connect()
    chunk_size = 10000
    
    with tqdm(total=len(features_df), desc="  Writing features") as pbar:
        for i in range(0, len(features_df), chunk_size):
            chunk = features_df.iloc[i:i+chunk_size]
            chunk.to_sql('engineered_features', engine, if_exists='append', index=False)
            pbar.update(len(chunk))
    
    logger.info(f"✓ Wrote {len(features_df):,} feature records to database")


def verify_features():
    """Verify engineered features in database"""
    logger.info("Verifying engineered features...")
    
    engine = db_manager.connect()
    
    # Count records
    count = pd.read_sql("SELECT COUNT(*) as count FROM engineered_features", engine).iloc[0]['count']
    logger.info(f"  Total records: {count:,}")
    
    # Check for nulls in key features
    null_check = pd.read_sql("""
        SELECT 
            SUM(CASE WHEN sales_lag_1 IS NULL THEN 1 ELSE 0 END) as null_lag1,
            SUM(CASE WHEN rolling_mean_4 IS NULL THEN 1 ELSE 0 END) as null_rolling,
            SUM(CASE WHEN temperature IS NULL THEN 1 ELSE 0 END) as null_temp
        FROM engineered_features
    """, engine)
    logger.info(f"  Null counts - lag_1: {null_check.iloc[0]['null_lag1']}, "
                f"rolling: {null_check.iloc[0]['null_rolling']}, "
                f"temp: {null_check.iloc[0]['null_temp']}")
    
    # Sample data
    sample = pd.read_sql("""
        SELECT * FROM engineered_features 
        WHERE sales_lag_1 IS NOT NULL 
        LIMIT 5
    """, engine)
    logger.info(f"  Sample record shape: {sample.shape}")
    logger.info(f"  Columns: {list(sample.columns)}")
    
    logger.info("✓ Feature verification complete")


def main():
    """Main feature engineering pipeline"""
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*60)
    
    try:
        # Step 1: Load data
        logger.info("\n[1/4] Loading data from database...")
        sales_df, features_df, stores_df = load_data_from_db()
        
        # Step 2: Engineer features
        logger.info("\n[2/4] Engineering features...")
        engineered_df = engineer_features(sales_df, features_df, stores_df)
        
        # Step 3: Write to database
        logger.info("\n[3/4] Writing to database...")
        write_features_to_db(engineered_df)
        
        # Step 4: Verify
        logger.info("\n[4/4] Verifying features...")
        verify_features()
        
        logger.info("\n" + "="*60)
        logger.info("✓ FEATURE ENGINEERING COMPLETE")
        logger.info("="*60)
        logger.info(f"Records processed: {len(engineered_df):,}")
        logger.info(f"Features created: {engineered_df.shape[1]} columns")
        
    except Exception as e:
        logger.error(f"\n✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
