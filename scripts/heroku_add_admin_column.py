#!/usr/bin/env python3
"""
Simple migration script to add is_admin column via Heroku.
This script uses environment variables directly.
"""

import os
import psycopg2
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def migrate_add_admin_column():
    """Add is_admin column to user table if it doesn't exist."""
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        logger.error("DATABASE_URL environment variable not found")
        return False
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Check if column already exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='user' AND column_name='is_admin'
        """)
        
        existing_columns = cur.fetchall()
        
        if existing_columns:
            logger.info("Column 'is_admin' already exists in user table")
            cur.close()
            conn.close()
            return True
        
        # Add the column with default value FALSE
        cur.execute('ALTER TABLE "user" ADD COLUMN is_admin BOOLEAN DEFAULT FALSE')
        conn.commit()
        
        logger.info("Successfully added 'is_admin' column to user table")
        
        # Count total users to determine if we need to set first user as admin
        cur.execute('SELECT COUNT(*) FROM "user"')
        user_count = cur.fetchone()[0]
        
        if user_count == 1:
            # If there's exactly one user, make them admin
            cur.execute('UPDATE "user" SET is_admin = TRUE WHERE id = (SELECT MIN(id) FROM "user")')
            conn.commit()
            logger.info("Set the first user as admin")
        elif user_count > 1:
            logger.warning(f"Found {user_count} existing users. No user automatically set as admin.")
            logger.warning("You may need to manually set an admin user.")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to add is_admin column: {e}")
        return False

if __name__ == "__main__":
    success = migrate_add_admin_column()
    if success:
        logger.info("Migration completed successfully")
        exit(0)
    else:
        logger.error("Migration failed")
        exit(1)