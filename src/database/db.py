"""
Core database connection and query functions.
"""
import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Get database path from environment or use default
DATABASE_PATH = os.getenv("DATABASE_PATH", "predict_presents.db")

@contextmanager
def get_db():
    """
    Context manager for database connections.

    Yields:
        sqlite3.Connection: Database connection with row factory set
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()

def init_database():
    """Initialize database with schema from schema.sql file."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = f.read()

    with get_db() as conn:
        conn.executescript(schema)

    logger.info("Database initialized successfully")

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
        cursor = conn.execute(query, params or ())
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
        cursor = conn.execute(query, params or ())
        if query.strip().upper().startswith("INSERT"):
            return cursor.lastrowid
        else:
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
        cursor = conn.executemany(query, params_list)
        return cursor.rowcount

def check_database_exists() -> bool:
    """Check if database file exists and has tables."""
    if not os.path.exists(DATABASE_PATH):
        return False

    try:
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = execute_query(query)
        return len(tables) >= 3  # We expect at least 3 tables
    except Exception:
        return False