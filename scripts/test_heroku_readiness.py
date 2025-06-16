#!/usr/bin/env python3
"""
Test script to verify Heroku readiness.
"""
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_database_switching():
    """Test that database switching works correctly."""
    print("Testing database switching...")
    
    # Test 1: Without DATABASE_URL (should use SQLite)
    if 'DATABASE_URL' in os.environ:
        del os.environ['DATABASE_URL']
    
    from src.database import db_factory
    db_module = db_factory.get_db_module()
    
    if db_module.__name__.endswith('.db'):
        print("[OK] Without DATABASE_URL: Using SQLite (correct)")
    else:
        print("[FAIL] Without DATABASE_URL: Not using SQLite (incorrect)")
    
    # Test 2: With DATABASE_URL (should use PostgreSQL)
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost/test'
    
    # Force reimport
    import importlib
    importlib.reload(db_factory)
    
    db_module = db_factory.get_db_module()
    
    if db_module.__name__.endswith('.db_postgres'):
        print("[OK] With DATABASE_URL: Using PostgreSQL (correct)")
    else:
        print("[FAIL] With DATABASE_URL: Not using PostgreSQL (incorrect)")
    
    # Cleanup
    if 'DATABASE_URL' in os.environ:
        del os.environ['DATABASE_URL']

def test_port_configuration():
    """Test that PORT environment variable works."""
    print("\nTesting PORT configuration...")
    
    # Test with PORT set
    os.environ['PORT'] = '5000'
    
    # Force reload of settings
    import importlib
    from src.config import settings as settings_module
    importlib.reload(settings_module)
    
    from src.config.settings import settings
    
    if settings.api.port == 5000:
        print("[OK] PORT environment variable respected (correct)")
    else:
        print(f"[FAIL] PORT not respected: got {settings.api.port} instead of 5000")
    
    # Cleanup
    del os.environ['PORT']

def test_imports():
    """Test that all critical imports work."""
    print("\nTesting critical imports...")
    
    try:
        import psycopg2
        print("[OK] psycopg2-binary installed")
    except ImportError:
        print("[FAIL] psycopg2-binary NOT installed")
    
    try:
        import gunicorn
        print("[OK] gunicorn installed")
    except ImportError:
        print("[FAIL] gunicorn NOT installed")
    
    try:
        from src.database.db_postgres import get_db
        print("[OK] PostgreSQL module imports correctly")
    except Exception as e:
        print(f"[FAIL] PostgreSQL module import failed: {e}")

def check_files():
    """Check that all required Heroku files exist."""
    print("\nChecking Heroku files...")
    
    files = {
        'Procfile': 'Web server configuration',
        'runtime.txt': 'Python version specification',
        'requirements.txt': 'Dependencies',
        'src/database/db_postgres.py': 'PostgreSQL support',
        'src/database/schema_postgres.sql': 'PostgreSQL schema',
        'src/database/db_factory.py': 'Database switching logic',
        'scripts/migrate_to_postgres.py': 'Migration script'
    }
    
    all_good = True
    for filepath, description in files.items():
        if os.path.exists(filepath):
            print(f"[OK] {filepath} - {description}")
        else:
            print(f"[FAIL] {filepath} - {description} (MISSING)")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("=== Heroku Readiness Test ===\n")
    
    check_files()
    test_imports()
    test_database_switching()
    test_port_configuration()
    
    print("\n=== Summary ===")
    print("If all tests passed [OK], your app is ready for Heroku deployment!")
    print("If any tests failed [FAIL], please fix the issues before deploying.")
    
    print("\nNext steps:")
    print("1. Create Heroku app: heroku create your-app-name")
    print("2. Add PostgreSQL: heroku addons:create heroku-postgresql:mini")
    print("3. Deploy: git push heroku main")
    print("4. Set environment variables as described in docs/heroku_deployment_steps.md")

if __name__ == "__main__":
    main()