#!/usr/bin/env python3
"""
Database initialization script for Heroku deployment.
This script reads the schema SQL file and executes it against the database.
"""

import os
import sys
import psycopg2
from pathlib import Path

def init_database():
    """Initialize the database schema on Heroku."""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not found")
        sys.exit(1)
    
    # Read the schema SQL file
    schema_file = Path(__file__).parent.parent / 'src' / 'database' / 'schema_postgres_init.sql'
    if not schema_file.exists():
        print(f"ERROR: Schema file not found: {schema_file}")
        sys.exit(1)
    
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        print(f"Read schema file: {schema_file}")
        print(f"Schema SQL length: {len(schema_sql)} characters")
        
        # Connect to database
        print("Connecting to database...")
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Execute schema SQL
        print("Executing schema initialization...")
        cursor.execute(schema_sql)
        
        print("✅ Database schema initialized successfully!")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"Created tables: {[table[0] for table in tables]}")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()