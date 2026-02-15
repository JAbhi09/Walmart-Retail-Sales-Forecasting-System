"""
Database Manager - Connection and utility functions
"""
import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'walmart_retail'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self.engine = None
        self.Session = None
    
    def get_connection_string(self, database=None):
        """Get PostgreSQL connection string"""
        from urllib.parse import quote_plus
        db_name = database or self.db_config['database']
        password = quote_plus(self.db_config['password'])
        return (
            f"postgresql://{self.db_config['user']}:{password}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{db_name}"
        )
    
    def create_database(self):
        """Create the main database if it doesn't exist"""
        # Connect to default 'postgres' database to create our database
        engine = create_engine(
            self.get_connection_string('postgres'),
            poolclass=NullPool,
            isolation_level='AUTOCOMMIT'
        )
        
        try:
            with engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text(f"SELECT 1 FROM pg_database WHERE datname = '{self.db_config['database']}'")
                )
                exists = result.fetchone()
                
                if not exists:
                    conn.execute(text(f"CREATE DATABASE {self.db_config['database']}"))
                    logger.info(f"✓ Created database: {self.db_config['database']}")
                else:
                    logger.info(f"✓ Database already exists: {self.db_config['database']}")
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
        finally:
            engine.dispose()
    
    def connect(self):
        """Create database engine and session"""
        if self.engine is None:
            self.engine = create_engine(
                self.get_connection_string(),
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )
            self.Session = sessionmaker(bind=self.engine)
            logger.info("✓ Database connection established")
        return self.engine
    
    def execute_schema(self, schema_file):
        """Execute SQL schema file"""
        if not os.path.exists(schema_file):
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        engine = self.connect()
        with engine.connect() as conn:
            # Execute schema (split by semicolon for multiple statements)
            for statement in schema_sql.split(';'):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            conn.commit()
        
        logger.info(f"✓ Schema executed from: {schema_file}")
    
    def get_session(self):
        """Get a new database session"""
        if self.Session is None:
            self.connect()
        return self.Session()
    
    def test_connection(self):
        """Test database connection"""
        try:
            engine = self.connect()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"✓ PostgreSQL version: {version[:50]}...")
                return True
        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("✓ Database connection closed")


# Singleton instance
db_manager = DatabaseManager()
