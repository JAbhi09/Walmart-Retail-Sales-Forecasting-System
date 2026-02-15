"""
Unit tests for database operations and connections.
"""
import pytest
import os
from sqlalchemy import text
from database.db_manager import DatabaseManager


class TestDatabaseConnection:
    """Test database connectivity and basic operations."""

    def test_database_connection(self, test_db_engine):
        """Test that database connection is successful."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_database_manager_initialization(self):
        """Test DatabaseManager can be initialized and connected."""
        db_manager = DatabaseManager()
        assert db_manager is not None
        # engine is None until .connect() is called
        assert db_manager.engine is None
        engine = db_manager.connect()
        assert db_manager.engine is not None
        assert engine is not None
        db_manager.close()

    def test_database_manager_test_connection(self):
        """Test DatabaseManager.test_connection() returns True."""
        db_manager = DatabaseManager()
        assert db_manager.test_connection() is True
        db_manager.close()

    def test_database_tables_exist(self, test_db_engine):
        """Test that required tables exist in database."""
        required_tables = [
            'stores',
            'raw_sales',
            'features',
            'engineered_features',
            'forecasts',
            'model_metadata'
        ]

        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            existing_tables = [row[0] for row in result]

        for table in required_tables:
            assert table in existing_tables, f"Table {table} does not exist"


class TestDatabaseOperations:
    """Test database CRUD operations."""

    def test_insert_store_data(self, test_db_engine):
        """Test inserting and cleaning up store data using a temp store_id."""
        temp_id = 99999  # high ID unlikely to conflict with real data

        with test_db_engine.connect() as conn:
            # Ensure clean slate
            conn.execute(text("DELETE FROM stores WHERE store_id = :sid"), {"sid": temp_id})
            conn.commit()

            # Insert a test row
            conn.execute(
                text("INSERT INTO stores (store_id, store_type, size) VALUES (:sid, :st, :sz)"),
                {"sid": temp_id, "st": "A", "sz": 100000},
            )
            conn.commit()

            # Verify insertion
            result = conn.execute(
                text("SELECT COUNT(*) FROM stores WHERE store_id = :sid"),
                {"sid": temp_id},
            )
            assert result.scalar() == 1

            # Clean up
            conn.execute(text("DELETE FROM stores WHERE store_id = :sid"), {"sid": temp_id})
            conn.commit()

    def test_query_sales_data(self, test_db_engine):
        """Test querying sales data."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM raw_sales"))
            count = result.scalar()

        assert count >= 0, "Sales table should be accessible"

    def test_database_indexes_exist(self, test_db_engine):
        """Test that performance indexes exist."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
            """))
            indexes = [row[0] for row in result]

        assert len(indexes) > 0, "No indexes found in database"


class TestDatabaseIntegrity:
    """Test database constraints and data integrity."""

    def test_foreign_key_constraints(self, test_db_engine):
        """Test that foreign key constraints are enforced."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*)
                FROM information_schema.table_constraints
                WHERE constraint_type = 'FOREIGN KEY'
            """))
            fk_count = result.scalar()

        assert fk_count > 0, "No foreign key constraints found"

    def test_primary_key_constraints(self, test_db_engine):
        """Test that primary key constraints exist."""
        with test_db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*)
                FROM information_schema.table_constraints
                WHERE constraint_type = 'PRIMARY KEY'
            """))
            pk_count = result.scalar()

        assert pk_count >= 6, f"Expected at least 6 primary keys, found {pk_count}"