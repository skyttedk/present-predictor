#!/usr/bin/env python3
"""
Migration script to add is_admin column to existing user table.
This script safely adds the is_admin column if it doesn't exist.
"""

import sys
import os
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import db_factory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def migrate_add_admin_column():
    """Add is_admin column to user table if it doesn't exist."""
    try:
        # Check if column already exists
        check_column_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='user' AND column_name='is_admin'
        """
        
        existing_columns = db_factory.execute_query(check_column_query)
        
        if existing_columns:
            logger.info("Column 'is_admin' already exists in user table")
            return True
        
        # Add the column with default value FALSE
        add_column_query = 'ALTER TABLE "user" ADD COLUMN is_admin BOOLEAN DEFAULT FALSE'
        db_factory.execute_write(add_column_query)
        
        logger.info("Successfully added 'is_admin' column to user table")
        
        # Count total users to determine if we need to set first user as admin
        count_query = 'SELECT COUNT(*) as count FROM "user"'
        count_result = db_factory.execute_query(count_query)
        user_count = count_result[0]['count'] if count_result else 0
        
        if user_count == 1:
            # If there's exactly one user, make them admin
            make_admin_query = 'UPDATE "user" SET is_admin = TRUE WHERE id = (SELECT MIN(id) FROM "user")'
            db_factory.execute_write(make_admin_query)
            logger.info("Set the first user as admin")
        elif user_count > 1:
            logger.warning(f"Found {user_count} existing users. No user automatically set as admin.")
            logger.warning("You may need to manually set an admin user using the CLI.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add is_admin column: {e}")
        return False

if __name__ == "__main__":
    success = migrate_add_admin_column()
    if success:
        logger.info("Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1)