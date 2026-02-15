"""
Database Setup Script
Creates databases, runs schema, and initializes MLflow tracking database
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import db_manager
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mlflow_database():
    """Create MLflow tracking database"""
    from urllib.parse import quote_plus
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    
    mlflow_db_name = 'mlflow_db'
    
    # URL-encode password for connection string
    password = quote_plus(db_config['password'])
    
    # Connect to postgres database to create mlflow_db
    engine = create_engine(
        f"postgresql://{db_config['user']}:{password}"
        f"@{db_config['host']}:{db_config['port']}/postgres",
        poolclass=NullPool,
        isolation_level='AUTOCOMMIT'
    )
    
    try:
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{mlflow_db_name}'")
            )
            exists = result.fetchone()
            
            if not exists:
                conn.execute(text(f"CREATE DATABASE {mlflow_db_name}"))
                logger.info(f"✓ Created MLflow database: {mlflow_db_name}")
            else:
                logger.info(f"✓ MLflow database already exists: {mlflow_db_name}")
    except Exception as e:
        logger.error(f"Error creating MLflow database: {e}")
        raise
    finally:
        engine.dispose()


def main():
    """Main setup function"""
    logger.info("="*60)
    logger.info("DATABASE SETUP")
    logger.info("="*60)
    
    try:
        # Step 1: Create main database
        logger.info("\n[1/5] Creating main database...")
        db_manager.create_database()
        
        # Step 2: Create MLflow database
        logger.info("\n[2/5] Creating MLflow database...")
        create_mlflow_database()
        
        # Step 3: Test connection
        logger.info("\n[3/5] Testing database connection...")
        if not db_manager.test_connection():
            raise Exception("Database connection test failed")
        
        # Step 4: Execute schema
        logger.info("\n[4/5] Creating database schema...")
        schema_file = project_root / 'database' / 'schema.sql'
        db_manager.execute_schema(str(schema_file))
        
        # Step 5: Verify tables
        logger.info("\n[5/5] Verifying tables...")
        engine = db_manager.connect()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            logger.info(f"✓ Created {len(tables)} tables: {', '.join(tables)}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ DATABASE SETUP COMPLETE")
        logger.info("="*60)
        logger.info(f"Main database: {os.getenv('DB_NAME')}")
        logger.info(f"MLflow database: mlflow_db")
        logger.info(f"Host: {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}")
        
    except Exception as e:
        logger.error(f"\n✗ Setup failed: {e}")
        sys.exit(1)
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
