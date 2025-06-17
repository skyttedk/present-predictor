#!/usr/bin/env python3
"""
Script to help migrate Heroku app to European region.
This script provides the commands needed for migration.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    response = input("Execute this command? (y/n): ")
    if response.lower() != 'y':
        print("Skipped.")
        return
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        sys.exit(1)

def main():
    """Main migration workflow."""
    current_app = "predict-presents-api"
    new_app = input(f"Enter new European app name (default: {current_app}-eu): ") or f"{current_app}-eu"
    
    print(f"\n🚀 Starting migration from {current_app} to {new_app}")
    
    # Step 1: Create new app
    run_command(
        f"heroku create {new_app} --region eu",
        "Creating new European app"
    )
    
    # Step 2: Add PostgreSQL
    run_command(
        f"heroku addons:create heroku-postgresql:mini --app {new_app}",
        "Adding PostgreSQL addon"
    )
    
    # Step 3: Copy environment variables
    print(f"\n📋 Current environment variables for {current_app}:")
    subprocess.run(f"heroku config --app {current_app}", shell=True)
    
    print(f"\n⚠️  Manually copy environment variables to {new_app} using:")
    print(f"heroku config:set VAR_NAME='value' --app {new_app}")
    input("Press Enter when environment variables are copied...")
    
    # Step 4: Add git remote
    run_command(
        f"git remote add eu https://git.heroku.com/{new_app}.git",
        "Adding git remote for European app"
    )
    
    # Step 5: Deploy
    run_command(
        "git push eu main",
        "Deploying code to European app"
    )
    
    # Step 6: Initialize database
    run_command(
        f"heroku run python scripts/init_heroku_db.py --app {new_app}",
        "Initializing database schema"
    )
    
    # Step 7: Create backup and restore
    print(f"\n💾 Data migration options:")
    print(f"1. Create fresh database with new users")
    print(f"2. Backup and restore existing data")
    
    choice = input("Choose option (1/2): ")
    
    if choice == "1":
        run_command(
            f"heroku run python scripts/create_test_api_user.py --app {new_app}",
            "Creating test user on new app"
        )
    elif choice == "2":
        run_command(
            f"heroku pg:backups:capture --app {current_app}",
            "Creating backup of current database"
        )
        
        print("\n📋 Get backup URL and restore manually:")
        print(f"heroku pg:backups:public-url --app {current_app}")
        print(f"heroku pg:backups:restore 'BACKUP_URL' DATABASE_URL --app {new_app}")
    
    # Step 8: Test new app
    print(f"\n🧪 Testing new European app:")
    print(f"heroku ps --app {new_app}")
    print(f"heroku logs --tail --app {new_app}")
    print(f"curl https://{new_app}.herokuapp.com/test")
    
    print(f"\n✅ Migration commands completed!")
    print(f"\n⚠️  Remember to:")
    print(f"1. Test the new app thoroughly")
    print(f"2. Update any external references to the new URL")
    print(f"3. Monitor for a few days before deleting the old app")
    print(f"4. Delete old app when confident: heroku apps:destroy {current_app}")

if __name__ == "__main__":
    main()