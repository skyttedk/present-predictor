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
    existing = db_factory.execute_query("SELECT id FROM user WHERE username = ?", (username,))
    if existing:
        raise ValueError(f"Username '{username}' already exists")

    api_key = generate_api_key()
    hashed_key = hash_api_key(api_key)

    query = "INSERT INTO user (username, api_key) VALUES (?, ?)"
    user_id = db_factory.execute_write(query, (username, hashed_key))

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
    users = db_factory.execute_query(query, (hashed_key,))

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

    return db_factory.execute_query(query)

def deactivate_user(username: str) -> bool:
    """
    Deactivate a user (soft delete).

    Args:
        username: Username to deactivate

    Returns:
        True if user was deactivated, False if user not found
    """
    query = "UPDATE user SET is_active = 0 WHERE username = ?"
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
    query = "UPDATE user SET is_active = 1 WHERE username = ?"
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
    users = db_factory.execute_query("SELECT id FROM user WHERE username = ?", (username,))
    if not users:
        return None

    new_api_key = generate_api_key()
    hashed_key = hash_api_key(new_api_key)

    query = "UPDATE user SET api_key = ? WHERE username = ?"
    affected = db_factory.execute_write(query, (hashed_key, username))

    if affected > 0:
        logger.info(f"Regenerated API key for user: {username}")
        return new_api_key
    return None