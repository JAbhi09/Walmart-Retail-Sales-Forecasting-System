"""
Data Loader - Load CSV files into PostgreSQL database
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
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_stores(data_dir):
    """Load stores.csv into stores table"""
    logger.info("Loading stores data...")
    
    # Read CSV
    stores_df = pd.read_csv(data_dir / 'stores.csv')
    logger.info(f"  Read {len(stores_df)} stores from CSV")
    
    # Rename columns to match database schema
    stores_df.columns = ['store_id', 'store_type', 'size']
    
    # Load to database
    engine = db_manager.connect()
    stores_df.to_sql('stores', engine, if_exists='append', index=False)
    
    logger.info(f"✓ Loaded {len(stores_df)} stores into database")
    return len(stores_df)


def load_sales(data_dir):
    """Load train.csv into raw_sales table"""
    logger.info("Loading sales data...")
    
    # Read CSV
    sales_df = pd.read_csv(data_dir / 'train.csv')
    logger.info(f"  Read {len(sales_df):,} sales records from CSV")
    
    # Rename columns to match database schema
    sales_df.columns = ['store_id', 'dept_id', 'date', 'weekly_sales', 'is_holiday']
    
    # Convert date to datetime
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Convert is_holiday to boolean
    sales_df['is_holiday'] = sales_df['is_holiday'].astype(bool)
    
    # Load to database in chunks for better performance
    engine = db_manager.connect()
    chunk_size = 10000
    
    with tqdm(total=len(sales_df), desc="  Loading sales") as pbar:
        for i in range(0, len(sales_df), chunk_size):
            chunk = sales_df.iloc[i:i+chunk_size]
            chunk.to_sql('raw_sales', engine, if_exists='append', index=False)
            pbar.update(len(chunk))
    
    logger.info(f"✓ Loaded {len(sales_df):,} sales records into database")
    return len(sales_df)


def load_features(data_dir):
    """Load features.csv into features table"""
    logger.info("Loading features data...")
    
    # Read CSV
    features_df = pd.read_csv(data_dir / 'features.csv')
    logger.info(f"  Read {len(features_df):,} feature records from CSV")
    
    # Rename columns to match database schema
    features_df.columns = [
        'store_id', 'date', 'temperature', 'fuel_price',
        'markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5',
        'cpi', 'unemployment', 'is_holiday'
    ]
    
    # Convert date to datetime
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    # Convert is_holiday to boolean
    features_df['is_holiday'] = features_df['is_holiday'].astype(bool)
    
    # Load to database in chunks
    engine = db_manager.connect()
    chunk_size = 5000
    
    with tqdm(total=len(features_df), desc="  Loading features") as pbar:
        for i in range(0, len(features_df), chunk_size):
            chunk = features_df.iloc[i:i+chunk_size]
            chunk.to_sql('features', engine, if_exists='append', index=False)
            pbar.update(len(chunk))
    
    logger.info(f"✓ Loaded {len(features_df):,} feature records into database")
    return len(features_df)


def verify_data():
    """Verify loaded data"""
    logger.info("Verifying loaded data...")
    
    engine = db_manager.connect()
    
    # Check stores
    stores_count = pd.read_sql("SELECT COUNT(*) as count FROM stores", engine).iloc[0]['count']
    logger.info(f"  Stores: {stores_count}")
    
    # Check sales
    sales_count = pd.read_sql("SELECT COUNT(*) as count FROM raw_sales", engine).iloc[0]['count']
    logger.info(f"  Sales records: {sales_count:,}")
    
    # Check features
    features_count = pd.read_sql("SELECT COUNT(*) as count FROM features", engine).iloc[0]['count']
    logger.info(f"  Feature records: {features_count:,}")
    
    # Date range
    date_range = pd.read_sql("""
        SELECT MIN(date) as min_date, MAX(date) as max_date 
        FROM raw_sales
    """, engine)
    logger.info(f"  Date range: {date_range.iloc[0]['min_date']} to {date_range.iloc[0]['max_date']}")
    
    # Departments
    dept_count = pd.read_sql("SELECT COUNT(DISTINCT dept_id) as count FROM raw_sales", engine).iloc[0]['count']
    logger.info(f"  Departments: {dept_count}")
    
    logger.info("✓ Data verification complete")


def main():
    """Main data loading function"""
    logger.info("="*60)
    logger.info("DATA LOADING PIPELINE")
    logger.info("="*60)
    
    data_dir = project_root / 'data'
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    try:
        # Connect to database
        logger.info("\n[1/5] Connecting to database...")
        db_manager.connect()
        
        # Load stores
        logger.info("\n[2/5] Loading stores...")
        load_stores(data_dir)
        
        # Load sales
        logger.info("\n[3/5] Loading sales...")
        load_sales(data_dir)
        
        # Load features
        logger.info("\n[4/5] Loading features...")
        load_features(data_dir)
        
        # Verify
        logger.info("\n[5/5] Verifying data...")
        verify_data()
        
        logger.info("\n" + "="*60)
        logger.info("✓ DATA LOADING COMPLETE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
