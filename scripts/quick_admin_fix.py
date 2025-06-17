#!/usr/bin/env python3
"""
Quick script to make user admin via direct SQL update.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def main():
    """Update user to admin via direct database query."""
    from database import db_factory
    
    print("=== Checking current users ===")
    users = db_factory.execute_query('SELECT id, username, is_admin, is_active FROM "user" ORDER BY id')
    
    if not users:
        print("No users found!")
        return
    
    for user in users:
        print(f"ID: {user['id']}, Username: {user['username']}, Admin: {user['is_admin']}, Active: {user['is_active']}")
    
    # Make first user admin
    first_user = users[0]
    username = first_user['username']
    
    if first_user['is_admin']:
        print(f"✅ User '{username}' is already admin!")
        return
    
    print(f"\n=== Making '{username}' admin ===")
    update_query = 'UPDATE "user" SET is_admin = true WHERE id = %s'
    affected = db_factory.execute_write(update_query, (first_user['id'],))
    
    if affected > 0:
        print(f"✅ Successfully made '{username}' admin!")
        
        # Verify
        updated_users = db_factory.execute_query('SELECT id, username, is_admin FROM "user" WHERE id = %s', (first_user['id'],))
        if updated_users:
            print(f"Verified - Admin status: {updated_users[0]['is_admin']}")
    else:
        print(f"❌ Failed to update user '{username}'")

if __name__ == "__main__":
    main()