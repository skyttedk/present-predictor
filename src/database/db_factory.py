"""
Database factory to switch between SQLite and PostgreSQL based on environment.
"""
import os
import logging

logger = logging.getLogger(__name__)

def get_db_module():
    """
    Returns the appropriate database module based on environment.
    
    Returns:
        module: Either db (SQLite) or db_postgres (PostgreSQL) module
    """
    if os.getenv("DATABASE_URL"):
        logger.info("Using PostgreSQL database (DATABASE_URL found)")
        from . import db_postgres as db_module
    else:
        logger.info("Using SQLite database (no DATABASE_URL)")
        from . import db as db_module
    
    return db_module

# Convenience functions that delegate to the appropriate module
def get_db():
    """Get database connection using appropriate backend."""
    return get_db_module().get_db()

def init_database():
    """Initialize database using appropriate backend."""
    return get_db_module().init_database()

def execute_query(query, params=None):
    """Execute query using appropriate backend."""
    return get_db_module().execute_query(query, params)

def execute_write(query, params=None):
    """Execute write operation using appropriate backend."""
    return get_db_module().execute_write(query, params)

def execute_many(query, params_list):
    """Execute many operations using appropriate backend."""
    return get_db_module().execute_many(query, params_list)

def check_database_exists():
    """Check if database exists using appropriate backend."""
    return get_db_module().check_database_exists()