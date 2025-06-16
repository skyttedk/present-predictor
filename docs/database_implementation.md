# Database Implementation Guide

This document contains the complete implementation for the SQLite database backend for the Predictive Gift Selection System.

## Directory Structure

```
src/database/
├── __init__.py
├── db.py              # Core database functions
├── schema.sql         # SQL schema file
├── users.py           # User-related queries
├── api_logs.py        # API logging functions
├── products.py        # Product cache functions
├── utils.py           # Helper functions
└── cli.py             # Command-line interface
```

## Implementation Files

### 1. Database Schema (`src/database/schema.sql`)

```sql
-- User table for API authentication
CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    api_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- API call logging
CREATE TABLE IF NOT EXISTS user_api_call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    api_route TEXT NOT NULL,
    request_payload TEXT,  -- JSON string
    response_payload TEXT, -- JSON string
    response_status_code INTEGER,
    response_time_ms REAL,
    error_message TEXT,
    FOREIGN KEY (user_id) REFERENCES user(id)
);

-- Product classification cache
CREATE TABLE IF NOT EXISTS product_attributes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL UNIQUE,
    product_hash TEXT NOT NULL UNIQUE,
    thread_id TEXT,
    run_id TEXT,
    item_main_category TEXT,
    item_sub_category TEXT,
    color TEXT,
    brand TEXT,
    vendor TEXT,
    value_price REAL,
    target_demographic TEXT,
    utility_type TEXT,
    durability TEXT,
    usage_type TEXT,
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    classification_status TEXT DEFAULT 'success',
    raw_description TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_api_key ON user(api_key);
CREATE INDEX IF NOT EXISTS idx_api_log_user_date ON user_api_call_log(user_id, date_time);
CREATE INDEX IF NOT EXISTS idx_api_log_route ON user_api_call_log(api_route);
CREATE INDEX IF NOT EXISTS idx_product_hash ON product_attributes(product_hash);
CREATE INDEX IF NOT EXISTS idx_product_id ON product_attributes(product_id);

-- Trigger to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_user_timestamp 
AFTER UPDATE ON user
BEGIN
    UPDATE user SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
```

### 2. Core Database Module (`src/database/db.py`)

```python
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
```

### 3. User Management Module (`src/database/users.py`)

```python
"""
User management functions for API authentication.
"""
import hashlib
import secrets
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging

from .db import execute_query, execute_write

logger = logging.getLogger(__name__)

def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage using SHA-256.
    
    Args:
        api_key: Plain text API key
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(api_key.encode()).hexdigest()

def create_user(username: str) -> Dict[str, str]:
    """
    Create a new user with API key.
    
    Args:
        username: Unique username
        
    Returns:
        Dictionary with user_id, username, and api_key
        
    Raises:
        ValueError: If username already exists
    """
    # Check if username already exists
    existing = execute_query("SELECT id FROM user WHERE username = ?", (username,))
    if existing:
        raise ValueError(f"Username '{username}' already exists")
    
    api_key = generate_api_key()
    hashed_key = hash_api_key(api_key)
    
    query = "INSERT INTO user (username, api_key) VALUES (?, ?)"
    user_id = execute_write(query, (username, hashed_key))
    
    logger.info(f"Created user: {username} (ID: {user_id})")
    
    return {
        "user_id": user_id,
        "username": username,
        "api_key": api_key  # Return unhashed key to user once
    }

def authenticate_user(api_key: str) -> Optional[Dict]:
    """
    Authenticate user by API key.
    
    Args:
        api_key: Plain text API key
        
    Returns:
        User dictionary if authenticated, None otherwise
    """
    hashed_key = hash_api_key(api_key)
    query = """
        SELECT id, username, is_active, created_at, updated_at
        FROM user 
        WHERE api_key = ? AND is_active = 1
    """
    users = execute_query(query, (hashed_key,))
    
    if users:
        user = users[0]
        logger.debug(f"Authenticated user: {user['username']}")
        return user
    
    logger.warning("Authentication failed for provided API key")
    return None

def list_users(active_only: bool = False) -> List[Dict]:
    """
    List all users.
    
    Args:
        active_only: If True, only return active users
        
    Returns:
        List of user dictionaries
    """
    query = """
        SELECT id, username, created_at, updated_at, is_active 
        FROM user
    """
    if active_only:
        query += " WHERE is_active = 1"
    query += " ORDER BY created_at DESC"
    
    return execute_query(query)

def deactivate_user(username: str) -> bool:
    """
    Deactivate a user (soft delete).
    
    Args:
        username: Username to deactivate
        
    Returns:
        True if user was deactivated, False if user not found
    """
    query = "UPDATE user SET is_active = 0 WHERE username = ?"
    affected = execute_write(query, (username,))
    
    if affected > 0:
        logger.info(f"Deactivated user: {username}")
        return True
    return False

def reactivate_user(username: str) -> bool:
    """
    Reactivate a previously deactivated user.
    
    Args:
        username: Username to reactivate
        
    Returns:
        True if user was reactivated, False if user not found
    """
    query = "UPDATE user SET is_active = 1 WHERE username = ?"
    affected = execute_write(query, (username,))
    
    if affected > 0:
        logger.info(f"Reactivated user: {username}")
        return True
    return False

def regenerate_api_key(username: str) -> Optional[str]:
    """
    Generate a new API key for a user.
    
    Args:
        username: Username to regenerate key for
        
    Returns:
        New API key if successful, None if user not found
    """
    # Check if user exists
    users = execute_query("SELECT id FROM user WHERE username = ?", (username,))
    if not users:
        return None
    
    new_api_key = generate_api_key()
    hashed_key = hash_api_key(new_api_key)
    
    query = "UPDATE user SET api_key = ? WHERE username = ?"
    affected = execute_write(query, (hashed_key, username))
    
    if affected > 0:
        logger.info(f"Regenerated API key for user: {username}")
        return new_api_key
    return None
```

### 4. API Logging Module (`src/database/api_logs.py`)

```python
"""
API call logging functions.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .db import execute_write, execute_query

logger = logging.getLogger(__name__)

def log_api_call(
    user_id: int,
    api_route: str,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    status_code: int,
    response_time_ms: float,
    error_message: Optional[str] = None
) -> int:
    """
    Log an API call to the database.
    
    Args:
        user_id: ID of the user making the request
        api_route: API endpoint route
        request_payload: Request body as dictionary
        response_payload: Response body as dictionary
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        error_message: Optional error message
        
    Returns:
        ID of the inserted log entry
    """
    query = """
        INSERT INTO user_api_call_log 
        (user_id, api_route, request_payload, response_payload, 
         response_status_code, response_time_ms, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    log_id = execute_write(query, (
        user_id,
        api_route,
        json.dumps(request_payload),
        json.dumps(response_payload),
        status_code,
        response_time_ms,
        error_message
    ))
    
    if status_code >= 400:
        logger.warning(f"API error logged - User: {user_id}, Route: {api_route}, Status: {status_code}")
    
    return log_id

def get_user_api_stats(user_id: int, days: int = 30) -> Dict[str, Any]:
    """
    Get API usage statistics for a user.
    
    Args:
        user_id: User ID
        days: Number of days to look back (default: 30)
        
    Returns:
        Dictionary with usage statistics
    """
    query = """
        SELECT 
            COUNT(*) as total_calls,
            AVG(response_time_ms) as avg_response_time,
            MAX(response_time_ms) as max_response_time,
            MIN(response_time_ms) as min_response_time,
            COUNT(CASE WHEN response_status_code >= 400 THEN 1 END) as error_count,
            COUNT(DISTINCT api_route) as unique_endpoints
        FROM user_api_call_log
        WHERE user_id = ? 
        AND date_time >= datetime('now', '-' || ? || ' days')
    """
    
    stats = execute_query(query, (user_id, days))[0]
    
    # Get calls by endpoint
    endpoint_query = """
        SELECT 
            api_route,
            COUNT(*) as call_count,
            AVG(response_time_ms) as avg_response_time
        FROM user_api_call_log
        WHERE user_id = ? 
        AND date_time >= datetime('now', '-' || ? || ' days')
        GROUP BY api_route
        ORDER BY call_count DESC
    """
    
    endpoints = execute_query(endpoint_query, (user_id, days))
    
    return {
        **stats,
        "endpoints": endpoints,
        "period_days": days
    }

def get_system_api_stats(hours: int = 24) -> Dict[str, Any]:
    """
    Get system-wide API statistics.
    
    Args:
        hours: Number of hours to look back (default: 24)
        
    Returns:
        Dictionary with system statistics
    """
    query = """
        SELECT 
            COUNT(*) as total_calls,
            COUNT(DISTINCT user_id) as active_users,
            AVG(response_time_ms) as avg_response_time,
            COUNT(CASE WHEN response_status_code >= 400 THEN 1 END) as error_count,
            COUNT(CASE WHEN response_status_code >= 500 THEN 1 END) as server_error_count
        FROM user_api_call_log
        WHERE date_time >= datetime('now', '-' || ? || ' hours')
    """
    
    stats = execute_query(query, (hours,))[0]
    
    # Get top users
    user_query = """
        SELECT 
            u.username,
            COUNT(*) as call_count
        FROM user_api_call_log l
        JOIN user u ON l.user_id = u.id
        WHERE l.date_time >= datetime('now', '-' || ? || ' hours')
        GROUP BY l.user_id
        ORDER BY call_count DESC
        LIMIT 10
    """
    
    top_users = execute_query(user_query, (hours,))
    
    return {
        **stats,
        "top_users": top_users,
        "period_hours": hours
    }

def get_recent_errors(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent API errors.
    
    Args:
        limit: Maximum number of errors to return
        
    Returns:
        List of error log entries
    """
    query = """
        SELECT 
            l.id,
            l.date_time,
            u.username,
            l.api_route,
            l.response_status_code,
            l.error_message,
            l.request_payload
        FROM user_api_call_log l
        JOIN user u ON l.user_id = u.id
        WHERE l.response_status_code >= 400
        ORDER BY l.date_time DESC
        LIMIT ?
    """
    
    errors = execute_query(query, (limit,))
    
    # Parse JSON payloads
    for error in errors:
        try:
            error['request_payload'] = json.loads(error['request_payload'])
        except:
            pass
    
    return errors

def cleanup_old_logs(days: int = 90) -> int:
    """
    Delete API logs older than specified days.
    
    Args:
        days: Number of days to keep logs
        
    Returns:
        Number of deleted records
    """
    query = """
        DELETE FROM user_api_call_log
        WHERE date_time < datetime('now', '-' || ? || ' days')
    """
    
    deleted = execute_write(query, (days,))
    logger.info(f"Cleaned up {deleted} old API log entries")
    
    return deleted
```

### 5. Product Classification Cache Module (`src/database/products.py`)

```python
"""
Product classification caching functions.
"""
import hashlib
import json
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import logging

from .db import execute_query, execute_write, execute_many

logger = logging.getLogger(__name__)

def get_product_hash(description: str) -> str:
    """
    Generate hash for product description.
    
    Args:
        description: Product description text
        
    Returns:
        MD5 hash of the description
    """
    return hashlib.md5(description.lower().strip().encode()).hexdigest()

def get_cached_classification(
    product_id: str, 
    description: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Check if product classification exists in cache.
    
    Args:
        product_id: Product identifier
        description: Product description (optional)
        
    Returns:
        Cached classification if found, None otherwise
    """
    if description:
        product_hash = get_product_hash(description)
        query = """
            SELECT * FROM product_attributes 
            WHERE product_id = ? OR product_hash = ?
            AND classification_status = 'success'
            ORDER BY classified_at DESC
            LIMIT 1
        """
        results = execute_query(query, (product_id, product_hash))
    else:
        query = """
            SELECT * FROM product_attributes 
            WHERE product_id = ?
            AND classification_status = 'success'
            ORDER BY classified_at DESC
            LIMIT 1
        """
        results = execute_query(query, (product_id,))
    
    if results:
        logger.debug(f"Cache hit for product: {product_id}")
        return results[0]
    
    logger.debug(f"Cache miss for product: {product_id}")
    return None

def cache_classification(
    product_id: str,
    description: str,
    thread_id: str,
    run_id: str,
    attributes: Dict[str, Any],
    status: str = 'success'
) -> int:
    """
    Cache product classification from OpenAI.
    
    Args:
        product_id: Product identifier
        description: Product description
        thread_id: OpenAI thread ID
        run_id: OpenAI run ID
        attributes: Classification attributes dictionary
        status: Classification status ('success', 'failed', 'partial')
        
    Returns:
        ID of the cached entry
    """
    product_hash = get_product_hash(description)
    
    query = """
        INSERT OR REPLACE INTO product_attributes
        (product_id, product_hash, thread_id, run_id, 
         item_main_category, item_sub_category, color, brand, vendor,
         value_price, target_demographic, utility_type, durability, usage_type,
         raw_description, classification_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    cache_id = execute_write(query, (
        product_id,
        product_hash,
        thread_id,
        run_id,
        attributes.get("itemMainCategory"),
        attributes.get("itemSubCategory"),
        attributes.get("color"),
        attributes.get("brand"),
        attributes.get("vendor"),
        attributes.get("valuePrice"),
        attributes.get("targetDemographic"),
        attributes.get("utilityType"),
        attributes.get("durability"),
        attributes.get("usageType"),
        description,
        status
    ))
    
    logger.info(f"Cached classification for product: {product_id}")
    return cache_id

def batch_cache_classifications(
    classifications: List[Dict[str, Any]]
) -> int:
    """
    Cache multiple product classifications at once.
    
    Args:
        classifications: List of classification dictionaries
        
    Returns:
        Number of cached entries
    """
    query = """
        INSERT OR REPLACE INTO product_attributes
        (product_id, product_hash, thread_id, run_id, 
         item_main_category, item_sub_category, color, brand, vendor,
         value_price, target_demographic, utility_type, durability, usage_type,
         raw_description, classification_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    params_list = []
    for item in classifications:
        product_hash = get_product_hash(item['description'])
        params = (
            item['product_id'],
            product_hash,
            item.get('thread_id'),
            item.get('run_id'),
            item['attributes'].get("itemMainCategory"),
            item['attributes'].get("itemSubCategory"),
            item['attributes'].get("color"),
            item['attributes'].get("brand"),
            item['attributes'].get("vendor"),
            item['attributes'].get("valuePrice"),
            item['attributes'].get("targetDemographic"),
            item['attributes'].get("utilityType"),
            item['attributes'].get("durability"),
            item['attributes'].get("usageType"),
            item['description'],
            item.get('status', 'success')
        )
        params_list.append(params)
    
    count = execute_many(query, params_list)
    logger.info(f"Batch cached {count} product classifications")
    return count

def get_classification_stats() -> Dict[str, Any]:
    """
    Get statistics about the classification cache.
    
    Returns:
        Dictionary with cache statistics
    """
    query = """
        SELECT 
            COUNT(*) as total_cached,
            COUNT(DISTINCT product_id) as unique_products,
            COUNT(CASE WHEN classification_status = 'success' THEN 1 END) as successful,
            COUNT(CASE WHEN classification_status = 'failed' THEN 1 END) as failed,
            MIN(classified_at) as oldest_entry,
            MAX(classified_at) as newest_entry
        FROM product_attributes
    """
    
    stats = execute_query(query)[0]
    
    # Get distribution by category
    category_query = """
        SELECT 
            item_main_category,
            COUNT(*) as count
        FROM product_attributes
        WHERE classification_status = 'success'
        AND item_main_category IS NOT NULL
        GROUP BY item_main_category
        ORDER BY count DESC
        LIMIT 10
    """
    
    categories = execute_query(category_query)
    
    return {
        **stats,
        "top_categories": categories
    }

def cleanup_old_cache(days: int = 30) -> int:
    """
    Remove old cached classifications.
    
    Args:
        days: Number of days to keep cache entries
        
    Returns:
        Number of deleted entries
    """
    query = """
        DELETE FROM product_attributes
        WHERE classified_at < datetime('now', '-' || ? || ' days')
    """
    
    deleted = execute_write(query, (days,))
    logger.info(f"Cleaned up {deleted} old cache entries")
    
    return deleted

def invalidate_product_cache(product_id: str) -> bool:
    """
    Invalidate cache for a specific product.
    
    Args:
        product_id: Product identifier
        
    Returns:
        True if cache was invalidated, False if product not found
    """
    query = """
        UPDATE product_attributes
        SET classification_status = 'invalidated'
        WHERE product_id = ?
    """
    
    affected = execute_write(query, (product_id,))
    
    if affected > 0:
        logger.info(f"Invalidated cache for product: {product_id}")
        return True
    return False
```

### 6. CLI Tool (`src/database/cli.py`)

```python
"""
Command-line interface for database management.
"""
import click
import sys
from tabulate import tabulate
from datetime import datetime

from .db import init_database, check_database_exists
from .users import (
    create_user, list_users, deactivate_user, 
    reactivate_user, regenerate_api_key
)
from .api_logs import (
    get_user_api_stats, get_system_api_stats, 
    cleanup_old_logs, get_recent_errors
)
from .products import (
    get_classification_stats, cleanup_old_cache
)

@click.group()
def cli():
    """Database management CLI for Predict Presents"""
    pass

@cli.command()
def init():
    """Initialize the database with schema"""
    if check_database_exists():
        if not click.confirm("Database already exists. Reinitialize?"):
            return
    
    try:
        init_database()
        click.echo("✅ Database initialized successfully")
    except Exception as e:
        click.echo(f"❌ Error initializing database: {e}", err=True)
        sys.exit(1)

@cli.group()
def user():
    """User management commands"""
    pass

@user.command('create')
@click.option('--username', required=True, help='Username for new user')
def create_api_user(username: str):
    """Create a new API user"""
    try:
        result = create_user(username)
        click.echo(f"\n✅ User created successfully!")
        click.echo(f"Username: {result['username']}")
        click.echo(f"API Key: {result['api_key']}")
        click.echo("\n⚠️  Save this API key - it won't be shown again!")
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

@user.command('list')
@click.option('--active-only', is_flag=True, help='Show only active users')
def list_api_users(active_only: bool):
    """List all API users"""
    users = list_users(active_only)
    
    if not users:
        click.echo("No users found")
        return
    
    headers = ['ID', 'Username', 'Status', 'Created', 'Updated']
    rows = []
    
    for user in users:
        rows.append([
            user['id'],
            user['username'],
            "Active" if user['is_active'] else "Inactive",
            user['created_at'][:19],  # Trim microseconds
            user['updated_at'][:19]
        ])
    
    click.echo(tabulate(rows, headers=headers, tablefmt='grid'))

@user.command('deactivate')
@click.option('--username', required=True, help='Username to deactivate')
def deactivate_api_user(username: str):
    """Deactivate a user"""
    if deactivate_user(username):
        click.echo(f"✅ User '{username}' deactivated")
    else:
        click.echo(f"❌ User '{username}' not found", err=True)

@user.command('reactivate')
@click.option('--username', required=True, help='Username to reactivate')
def reactivate_api_user(username: str):
    """Reactivate a user"""
    if reactivate_user(username):
        click.echo(f"✅ User '{username}' reactivated")
    else:
        click.echo(f"❌ User '{username}' not found", err=True)

@user.command('regenerate-key')
@click.option('--username', required=True, help='Username to regenerate key for')
def regenerate_key(username: str):
    """Generate a new API key for a user"""
    if not click.confirm(f"Generate new API key for '{username}'?"):
        return
    
    new_key = regenerate_api_key(username)
    if new_key:
        click.echo(f"\n✅ New API key generated!")
        click.echo(f"Username: {username}")
        click.echo(f"New API Key: {new_key}")
        click.echo("\n⚠️  Save this API key - it won't be shown again!")
    else:
        click.echo(f"❌ User '{username}' not found", err=True)

@cli.group()
def stats():
    """View statistics and logs"""
    pass

@stats.command('user')
@click.option('--username', required=True, help='Username to get stats for')
@click.option('--days', default=30, help='Number of days to look back')
def user_stats(username: str, days: int):
    """Show API usage stats for a user"""
    # Get user ID
    users = list_users()
    user = next((u for u in users if u['username'] == username), None)
    
    if not user:
        click.echo(f"❌ User '{username}' not found", err=True)
        return
    
    stats = get_user_api_stats(user['id'], days)
    
    click.echo(f"\n📊 API Usage Stats for {username} (last {days} days)")
    click.echo(f"Total Calls: {stats['total_calls']}")
    click.echo(f"Error Count: {stats['error_count']}")
    click.echo(f"Avg Response Time: {stats['avg_response_time']:.2f}ms")
    click.echo(f"Unique Endpoints: {stats['unique_endpoints']}")
    
    if stats['endpoints']:
        click.echo("\nTop Endpoints:")
        headers = ['Endpoint', 'Calls', 'Avg Time (ms)']
        rows = [[e['api_route'], e['call_count'], f"{e['avg_response_time']:.2f}"] 
                for e in stats['endpoints'][:5]]
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))

@stats.command('system')
@click.option('--hours', default=24, help='Number of hours to look back')
def system_stats(hours: int):
    """Show system-wide API stats"""
    stats = get_system_api_stats(hours)
    
    click.echo(f"\n📊 System API Stats (last {hours} hours)")
    click.echo(f"Total Calls: {stats['total_calls']}")
    click.echo(f"Active Users: {stats['active_users']}")
    click.echo(f"Error Count: {stats['error_count']} ({stats['server_error_count']} server errors)")
    click.echo(f"Avg Response Time: {stats['avg_response_time']:.2f}ms")
    
    if stats['top_users']:
        click.echo("\nTop Users:")
        headers = ['Username', 'Calls']
        rows = [[u['username'], u['call_count']] for u in stats['top_users']]
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))

@stats.command('errors')
@click.option('--limit', default=10, help='Number of errors to show')
def recent_errors(limit: int):
    """Show recent API errors"""
    errors = get_recent_errors(limit)
    
    if not errors:
        click.echo("No recent errors found")
        return
    
    click.echo(f"\n❌ Recent API Errors (showing {len(errors)})")
    
    for error in errors:
        click.echo(f"\n[{error['date_time']}] {error['username']} - {error['api_route']}")
        click.echo(f"Status: {error['response_status_code']}")
        if error['error_message']:
            click.echo(f"Error: {error['error_message']}")

@stats.command('cache')
def cache_stats():
    """Show product classification cache stats"""
    stats = get_classification_stats()
    
    click.echo("\n📦 Product Classification Cache Stats")
    click.echo(f"Total Cached: {stats['total_cached']}")
    click.echo(f"Unique Products: {stats['unique_products']}")
    click.echo(f"Successful: {stats['successful']}")
    click.echo(f"Failed: {stats['failed']}")
    
    if stats['oldest_entry']:
        click.echo(f"Oldest Entry: {stats['oldest_entry']}")
        click.echo(f"Newest Entry: {stats['newest_entry']}")
    
    if stats['top_categories']:
        click.echo("\nTop Categories:")
        headers = ['Category', 'Count']
        rows = [[c['item_main_category'], c['count']] for c in stats['top_categories']]
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))

@cli.group()
def cleanup():
    """Database cleanup commands"""
    pass

@cleanup.command('logs')
@click.option('--days', default=90, help='Keep logs from last N days')
def cleanup_logs(days: int):
    """Clean up old API logs"""
    if not click.confirm(f"Delete API logs older than {days} days?"):
        return
    
    deleted = cleanup_old_logs(days)
    click.echo(f"✅ Deleted {deleted} old log entries")

@cleanup.command('cache')
@click.option('--days', default=30, help='Keep cache entries from last N days')
def cleanup_cache(days: int):
    """Clean up old product cache entries"""
    if not click.confirm(f"Delete cache entries older than {days} days?"):
        return
    
    deleted = cleanup_old_cache(days)
    click.echo(f"✅ Deleted {deleted} old cache entries")

if __name__ == '__main__':
    cli()
```

### 7. Database Module Init (`src/database/__init__.py`)

```python
"""
Database module for the Predictive Gift Selection System.

This module provides SQLite database functionality for:
- User authentication and API key management
- API call logging and analytics
- Product classification caching
"""

from .db import init_database, check_database_exists, get_db
from .users import create_user, authenticate_user, list_users
from .api_logs import log_api_call, get_user_api_stats
from .products import get_cached_classification, cache_classification

__all__ = [
    'init_database',
    'check_database_exists',
    'get_db',
    'create_user',
    'authenticate_user',
    'list_users',
    'log_api_call',
    'get_user_api_stats',
    'get_cached_classification',
    'cache_classification',
]
```

## Usage Examples

### Initialize Database
```bash
python -m src.database.cli init
```

### Create API User
```bash
python -m src.database.cli user create --username "client1"
```

### View Statistics
```bash
python -m src.database.cli stats system --hours 24
python -m src.database.cli stats user --username "client1" --days 7
```

### API Integration Example
```python
from fastapi import Depends, HTTPException, Header
from src.database.users import authenticate_user
from src.database.api_logs import log_api_call
import time

async def get_current_user(api_key: str = Header(...)):
    """Dependency for API authentication"""
    user = authenticate_user(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user

@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log API calls"""
    start_time = time.time()
    response = await call_next(request)
    response_time = (time.time() - start_time) * 1000
    
    # Log to database (simplified - would need user context)
    # This is a conceptual example
    return response
```

## Testing

Create test file `tests/test_database.py`:

```python
import pytest
import tempfile
import os
from src.database.db import init_database, check_database_exists
from src.database.users import create_user, authenticate_user
from src.database.products import cache_classification, get_cached_classification

@pytest.fixture
def test_db():
    """Create temporary test database"""
    fd, path = tempfile.mkstemp()
    os.environ["DATABASE_PATH"] = path
    init_database()
    yield path
    os.unlink(path)

def test_database_init(test_db):
    """Test database initialization"""
    assert check_database_exists()

def test_user_creation(test_db):
    """Test user creation and authentication"""
    result = create_user("testuser")
    assert result["username"] == "testuser"
    assert len(result["api_key"]) > 0
    
    # Test authentication
    user = authenticate_user(result["api_key"])
    assert user is not None
    assert user["username"] == "testuser"
    
    # Test invalid key
    assert authenticate_user("invalid_key") is None

def test_product_cache(test_db):
    """Test product classification caching"""
    attributes = {
        "itemMainCategory": "Electronics",
        "itemSubCategory": "Headphones",
        "brand": "Sony",
        "targetDemographic": "unisex"
    }
    
    # Cache classification
    cache_classification(
        product_id="TEST123",
        description="Sony Wireless Headphones",
        thread_id="thread_abc",
        run_id="run_xyz",
        attributes=attributes
    )
    
    # Retrieve from cache
    cached = get_cached_classification("TEST123")
    assert cached is not None
    assert cached["item_main_category"] == "Electronics"
    assert cached["brand"] == "Sony"
```

## Next Steps

1. Add this code to requirements.txt:
```
tabulate>=0.9.0  # For CLI table formatting
click>=8.0.0     # For CLI commands
```

2. Switch to Code mode to implement these files

3. Initialize the database and create first user

4. Integrate with FastAPI for authentication and logging

5. Add caching to the OpenAI classification pipeline