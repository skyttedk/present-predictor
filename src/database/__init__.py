"""
Database module for the Predictive Gift Selection System.

This module provides SQLite database functionality for:
- User authentication and API key management
- API call logging and analytics
- Product classification caching
"""

# Ensure src directory is in Python path for relative imports
# This might be needed if running cli.py directly for testing,
# but generally, it's better to run as a module: python -m src.database.cli
try:
    from .db import init_database, check_database_exists, get_db
    from .users import create_user, authenticate_user, list_users, deactivate_user, reactivate_user, regenerate_api_key
    from .api_logs import log_api_call, get_user_api_stats, get_system_api_stats, cleanup_old_logs, get_recent_errors
    from .presents import get_cached_present_classification, cache_present_classification, batch_cache_present_classifications, get_present_classification_stats, cleanup_old_present_cache, invalidate_present_cache
except ImportError:
    # This block allows running the script directly for development,
    # assuming it's in src/database/
    import os
    import sys
    if os.path.basename(os.getcwd()) == "database" and \
       os.path.basename(os.path.dirname(os.getcwd())) == "src":
        sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
    from src.database.db import init_database, check_database_exists, get_db
    from src.database.users import create_user, authenticate_user, list_users, deactivate_user, reactivate_user, regenerate_api_key
    from src.database.api_logs import log_api_call, get_user_api_stats, get_system_api_stats, cleanup_old_logs, get_recent_errors
    from src.database.presents import get_cached_present_classification, cache_present_classification, batch_cache_present_classifications, get_present_classification_stats, cleanup_old_present_cache, invalidate_present_cache


__all__ = [
    'init_database',
    'check_database_exists',
    'get_db',
    'create_user',
    'authenticate_user',
    'list_users',
    'deactivate_user',
    'reactivate_user',
    'regenerate_api_key',
    'log_api_call',
    'get_user_api_stats',
    'get_system_api_stats',
    'cleanup_old_logs',
    'get_recent_errors',
    'get_cached_present_classification', # Signature changed
    'cache_present_classification',
    'batch_cache_present_classifications',
    'get_present_classification_stats',
    'cleanup_old_present_cache',
    'invalidate_present_cache', # Signature changed
]