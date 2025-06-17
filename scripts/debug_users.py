#!/usr/bin/env python3
"""
Debug script to check user database state.
"""

import os
import psycopg2
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_users():
    """Check the current state of users in the database."""
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        logger.error("DATABASE_URL environment variable not found")
        return False
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Check users table structure
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name='user' 
            ORDER BY ordinal_position
        """)
        
        columns = cur.fetchall()
        logger.info("User table structure:")
        for col_name, col_type in columns:
            logger.info(f"  {col_name}: {col_type}")
        
        # Check all users
        cur.execute('SELECT id, username, is_admin, is_active FROM "user" ORDER BY id')
        users = cur.fetchall()
        logger.info(f"Found {len(users)} users:")
        for user in users:
            logger.info(f"  ID: {user[0]}, Username: {user[1]}, Is Admin: {user[2]}, Is Active: {user[3]}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to debug users: {e}")
        return False

if __name__ == "__main__":
    success = debug_users()
    if success:
        logger.info("Debug completed successfully")
        exit(0)
    else:
        logger.error("Debug failed")
        exit(1)