"""
User management functions for API authentication.
"""
import hashlib
import secrets
from typing import Optional, Dict, List
import logging

from . import db_factory

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

def count_users() -> int:
    """
    Count total number of users in the system.
    
    Returns:
        Total number of users
    """
    query = 'SELECT COUNT(*) as count FROM "user"'
    result = db_factory.execute_query(query)
    return result[0]['count'] if result else 0

def create_user(username: str, is_admin: bool = None) -> Dict[str, str]:
    """
    Create a new user with API key.
    
    Args:
        username: Unique username
        is_admin: Whether user should be admin. If None, first user becomes admin automatically.

    Returns:
        Dictionary with user_id, username, api_key, and is_admin

    Raises:
        ValueError: If username already exists
    """
    # Check if username already exists
    existing = db_factory.execute_query('SELECT id FROM "user" WHERE username = %s', (username,))
    if existing:
        raise ValueError(f"Username '{username}' already exists")

    # Determine admin status
    if is_admin is None:
        # First user automatically becomes admin
        user_count = count_users()
        is_admin = (user_count == 0)
    
    api_key = generate_api_key()
    hashed_key = hash_api_key(api_key)

    query = 'INSERT INTO "user" (username, api_key, is_admin) VALUES (%s, %s, %s)'
    user_id = db_factory.execute_write(query, (username, hashed_key, is_admin))

    logger.info(f"Created user: {username} (ID: {user_id}, Admin: {is_admin})")

    return {
        "user_id": user_id,
        "username": username,
        "api_key": api_key,  # Return unhashed key to user once
        "is_admin": is_admin
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
        SELECT id, username, is_active, is_admin, created_at, updated_at
        FROM "user"
        WHERE api_key = %s AND is_active = true
    """
    users = db_factory.execute_query(query, (hashed_key,))

    if users:
        user = users[0]
        logger.debug(f"Authenticated user: {user['username']} (Admin: {user['is_admin']})")
        return user

    logger.warning("Authentication failed for provided API key")
    return None

def is_admin_user(api_key: str) -> bool:
    """
    Check if the user with given API key is an admin.
    
    Args:
        api_key: Plain text API key
        
    Returns:
        True if user is admin and active, False otherwise
    """
    user = authenticate_user(api_key)
    return user is not None and user.get('is_admin', False)

def list_users(active_only: bool = False) -> List[Dict]:
    """
    List all users.

    Args:
        active_only: If True, only return active users

    Returns:
        List of user dictionaries
    """
    query = """
        SELECT id, username, created_at, updated_at, is_active, is_admin
        FROM "user"
    """
    if active_only:
        query += " WHERE is_active = true"
    query += " ORDER BY created_at DESC"

    return db_factory.execute_query(query)

def delete_user(username: str) -> bool:
    """
    Permanently delete a user from the database.

    Args:
        username: Username to delete

    Returns:
        True if user was deleted, False if user not found
    """
    # Check if user exists first
    existing = db_factory.execute_query('SELECT id FROM "user" WHERE username = %s', (username,))
    if not existing:
        return False
    
    user_id = existing[0]['id']
    
    # Delete related API call logs first (due to foreign key constraint)
    db_factory.execute_write('DELETE FROM user_api_call_log WHERE user_id = %s', (user_id,))
    
    # Delete the user
    affected = db_factory.execute_write('DELETE FROM "user" WHERE username = %s', (username,))

    if affected > 0:
        logger.info(f"Deleted user: {username} (ID: {user_id})")
        return True
    return False

def deactivate_user(username: str) -> bool:
    """
    Deactivate a user (soft delete).

    Args:
        username: Username to deactivate

    Returns:
        True if user was deactivated, False if user not found
    """
    query = 'UPDATE "user" SET is_active = false WHERE username = %s'
    affected = db_factory.execute_write(query, (username,))

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
    query = 'UPDATE "user" SET is_active = true WHERE username = %s'
    affected = db_factory.execute_write(query, (username,))

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
    users = db_factory.execute_query('SELECT id FROM "user" WHERE username = %s', (username,))
    if not users:
        return None

    new_api_key = generate_api_key()
    hashed_key = hash_api_key(new_api_key)

    query = 'UPDATE "user" SET api_key = %s WHERE username = %s'
    affected = db_factory.execute_write(query, (hashed_key, username))

    if affected > 0:
        logger.info(f"Regenerated API key for user: {username}")
        return new_api_key
    return None