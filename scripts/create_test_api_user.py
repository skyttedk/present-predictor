#!/usr/bin/env python3
"""
Script to create a test user via direct database access and get their API key.
"""

import os
import psycopg2
import sys
sys.path.insert(0, '.')
from src.database.users import create_user
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_user():
    """Create a test user and return their API key."""
    try:
        # Create a new user (will automatically be non-admin since first user already exists)
        result = create_user('api_test_user')
        logger.info(f"Created user successfully:")
        logger.info(f"  ID: {result['user_id']}")
        logger.info(f"  Username: {result['username']}")
        logger.info(f"  Is Admin: {result['is_admin']}")
        logger.info(f"  API Key: {result['api_key']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create test user: {e}")
        return False

if __name__ == "__main__":
    success = create_test_user()
    if success:
        logger.info("User creation completed successfully")
        exit(0)
    else:
        logger.error("User creation failed")
        exit(1)