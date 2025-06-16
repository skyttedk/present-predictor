"""
PostgreSQL database connection and query functions for Heroku deployment.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL from environment, handle Heroku's postgres:// -> postgresql:// issue."""
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    return database_url

@contextmanager
def get_db():
    """
    Context manager for PostgreSQL database connections.

    Yields:
        psycopg2.connection: Database connection with RealDictCursor
    """
    database_url = get_database_url()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    parsed = urlparse(database_url)
    
    try:
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path[1:],  # Remove leading slash
            cursor_factory=RealDictCursor
        )
        yield conn
        conn.commit()
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def init_database():
    """Initialize database with PostgreSQL schema only if tables don't exist."""
    # Use the safe initialization schema that doesn't drop existing tables
    schema_path = os.path.join(os.path.dirname(__file__), "schema_postgres_init.sql")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = f.read()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(schema)

    logger.info("PostgreSQL database initialized successfully")

def execute_query(query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """
    Execute a SELECT query and return results as list of dicts.

    Args:
        query: SQL query string
        params: Query parameters as tuple

    Returns:
        List of dictionaries representing rows
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        return [dict(row) for row in cursor.fetchall()]

def execute_write(query: str, params: Optional[tuple] = None) -> int:
    """
    Execute INSERT/UPDATE/DELETE and return lastrowid or rowcount.

    Args:
        query: SQL query string
        params: Query parameters as tuple

    Returns:
        Last inserted row ID for INSERT, or affected row count for UPDATE/DELETE
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # PostgreSQL: All INSERT queries should use RETURNING id
        if query.strip().upper().startswith("INSERT"):
            # Add RETURNING id if not already present
            if "RETURNING" not in query.upper():
                query = query.rstrip(';') + " RETURNING id"
            
            cursor.execute(query, params or ())
            result = cursor.fetchone()
            return result['id'] if result else 0
        else:
            cursor.execute(query, params or ())
            return cursor.rowcount

def execute_many(query: str, params_list: List[tuple]) -> int:
    """
    Execute multiple INSERT/UPDATE/DELETE operations.

    Args:
        query: SQL query string
        params_list: List of parameter tuples

    Returns:
        Number of affected rows
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        return cursor.rowcount

def check_database_exists() -> bool:
    """Check if database tables exist."""
    database_url = get_database_url()
    if not database_url:
        return False

    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        tables = execute_query(query)
        # We expect at least 3 tables (user, user_api_call_log, present_attributes)
        return len(tables) >= 3
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False