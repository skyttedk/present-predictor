#!/usr/bin/env python3
"""
Create a test user for Heroku deployment.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, '/app/src')

from database.users import create_user

def main():
    """Create a test user and print the API key."""
    try:
        username = 'heroku-test-user'
        api_key = create_user(username)
        print('✅ User created successfully!')
        print(f'Username: {username}')
        print(f'API Key: {api_key}')
        print('')
        print('You can now test the API with:')
        print(f'curl -H "X-API-Key: {api_key}" https://predict-presents-api-b58dda526ddb.herokuapp.com/test')
        
    except Exception as e:
        print(f'❌ Error creating user: {e}')
        sys.exit(1)

if __name__ == "__main__":
    main()