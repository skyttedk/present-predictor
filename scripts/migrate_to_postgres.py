#!/usr/bin/env python3
"""
Migration script to transfer data from SQLite to PostgreSQL for Heroku deployment.
"""
import os
import sys
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
import logging
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_postgres_connection(database_url):
    """Create PostgreSQL connection from DATABASE_URL."""
    # Handle Heroku's postgres:// -> postgresql:// issue
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    parsed = urlparse(database_url)
    
    return psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port,
        user=parsed.username,
        password=parsed.password,
        database=parsed.path[1:],
        cursor_factory=RealDictCursor
    )

def get_sqlite_connection(db_path):
    """Create SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def create_postgres_schema(pg_conn):
    """Create PostgreSQL schema."""
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'database', 'schema_postgres.sql')
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema = f.read()
    
    cursor = pg_conn.cursor()
    cursor.execute(schema)
    pg_conn.commit()
    logger.info("PostgreSQL schema created successfully")

def migrate_table(sqlite_conn, pg_conn, table_name, column_mappings=None):
    """
    Migrate data from SQLite table to PostgreSQL.
    
    Args:
        sqlite_conn: SQLite connection
        pg_conn: PostgreSQL connection
        table_name: Name of the table to migrate
        column_mappings: Optional dict to rename columns during migration
    """
    # Get data from SQLite
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute(f"SELECT * FROM {table_name}")
    rows = sqlite_cursor.fetchall()
    
    if not rows:
        logger.info(f"No data to migrate in table: {table_name}")
        return
    
    # Get column names
    columns = [desc[0] for desc in sqlite_cursor.description]
    
    # Apply column mappings if provided
    if column_mappings:
        columns = [column_mappings.get(col, col) for col in columns]
    
    # Prepare PostgreSQL insert
    pg_cursor = pg_conn.cursor()
    
    # Handle special table names that need quotes
    pg_table_name = f'"{table_name}"' if table_name == "user" else table_name
    
    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)
    
    insert_query = f"INSERT INTO {pg_table_name} ({columns_str}) VALUES ({placeholders})"
    
    # Migrate data
    count = 0
    for row in rows:
        try:
            # Convert row to list and handle any necessary type conversions
            values = []
            for i, value in enumerate(row):
                # SQLite stores booleans as 0/1, PostgreSQL expects true/false
                if isinstance(value, int) and columns[i] in ['is_active']:
                    value = bool(value)
                values.append(value)
            
            pg_cursor.execute(insert_query, values)
            count += 1
        except Exception as e:
            logger.error(f"Error migrating row in {table_name}: {e}")
            logger.error(f"Row data: {dict(row)}")
            raise
    
    pg_conn.commit()
    logger.info(f"Migrated {count} rows to {table_name}")

def reset_sequences(pg_conn):
    """Reset PostgreSQL sequences to match the data."""
    cursor = pg_conn.cursor()
    
    sequences = [
        ('user', 'user_id_seq'),
        ('user_api_call_log', 'user_api_call_log_id_seq'),
        ('present_attributes', 'present_attributes_id_seq')
    ]
    
    for table, sequence in sequences:
        # Handle special table names
        pg_table = f'"{table}"' if table == "user" else table
        
        cursor.execute(f"""
            SELECT setval('{sequence}', COALESCE(MAX(id), 1)) 
            FROM {pg_table}
        """)
    
    pg_conn.commit()
    logger.info("PostgreSQL sequences reset successfully")

def migrate_all(sqlite_path, postgres_url):
    """Perform complete migration from SQLite to PostgreSQL."""
    logger.info("Starting migration from SQLite to PostgreSQL...")
    
    # Connect to databases
    sqlite_conn = get_sqlite_connection(sqlite_path)
    pg_conn = get_postgres_connection(postgres_url)
    
    try:
        # Create schema
        create_postgres_schema(pg_conn)
        
        # Migrate tables in order (respecting foreign keys)
        tables = [
            ('user', None),
            ('user_api_call_log', None),
            ('present_attributes', None)
        ]
        
        for table_name, mappings in tables:
            logger.info(f"Migrating table: {table_name}")
            migrate_table(sqlite_conn, pg_conn, table_name, mappings)
        
        # Reset sequences
        reset_sequences(pg_conn)
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        pg_conn.rollback()
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()

def verify_migration(sqlite_path, postgres_url):
    """Verify that migration was successful by comparing row counts."""
    logger.info("Verifying migration...")
    
    sqlite_conn = get_sqlite_connection(sqlite_path)
    pg_conn = get_postgres_connection(postgres_url)
    
    try:
        tables = ['user', 'user_api_call_log', 'present_attributes']
        
        for table in tables:
            # SQLite count
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table}")
            sqlite_count = sqlite_cursor.fetchone()[0]
            
            # PostgreSQL count
            pg_cursor = pg_conn.cursor()
            pg_table = f'"{table}"' if table == "user" else table
            pg_cursor.execute(f"SELECT COUNT(*) FROM {pg_table}")
            pg_count = pg_cursor.fetchone()['count']
            
            if sqlite_count == pg_count:
                logger.info(f"✓ {table}: {sqlite_count} rows")
            else:
                logger.error(f"✗ {table}: SQLite={sqlite_count}, PostgreSQL={pg_count}")
    
    finally:
        sqlite_conn.close()
        pg_conn.close()

def main():
    """Main migration function."""
    # Check for required environment variable
    postgres_url = os.getenv('DATABASE_URL')
    if not postgres_url:
        logger.error("DATABASE_URL environment variable not set!")
        logger.info("Usage: DATABASE_URL=postgresql://... python scripts/migrate_to_postgres.py")
        sys.exit(1)
    
    # Default SQLite path
    sqlite_path = os.getenv('SQLITE_PATH', 'predict_presents.db')
    
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite database not found: {sqlite_path}")
        sys.exit(1)
    
    # Perform migration
    try:
        migrate_all(sqlite_path, postgres_url)
        verify_migration(sqlite_path, postgres_url)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()