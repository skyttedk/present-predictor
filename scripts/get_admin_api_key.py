#!/usr/bin/env python3
"""
Script to regenerate admin API key for testing purposes.
"""

import os
import sys
sys.path.insert(0, '.')
from src.database.users import regenerate_api_key
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_admin_api_key():
    """Regenerate API key for admin user."""
    try:
        # Regenerate API key for the admin user
        new_api_key = regenerate_api_key('heroku-test-user')
        if new_api_key:
            logger.info(f"Successfully regenerated API key for admin user 'heroku-test-user':")
            logger.info(f"  New API Key: {new_api_key}")
            return True
        else:
            logger.error("Failed to regenerate API key - user not found")
            return False
        
    except Exception as e:
        logger.error(f"Failed to regenerate admin API key: {e}")
        return False

if __name__ == "__main__":
    success = get_admin_api_key()
    if success:
        logger.info("API key regeneration completed successfully")
        exit(0)
    else:
        logger.error("API key regeneration failed")
        exit(1)