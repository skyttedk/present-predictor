"""
Command-line interface for database management.
"""
import click
import sys
from tabulate import tabulate
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

# Ensure src directory is in Python path for relative imports
# This might be needed if running cli.py directly for testing,
# but generally, it's better to run as a module: python -m src.database.cli
try:
    from .db import init_database, check_database_exists, get_db
    from .users import (
        create_user, list_users, deactivate_user,
        reactivate_user, regenerate_api_key
    )
    from .api_logs import (
        get_user_api_stats, get_system_api_stats,
        cleanup_old_logs, get_recent_errors
    )
    from .presents import (
        get_present_classification_stats, cleanup_old_present_cache
    )
except ImportError:
    # This block allows running the script directly for development,
    # assuming it's in src/database/
    import os
    if os.path.basename(os.getcwd()) == "database" and \
       os.path.basename(os.path.dirname(os.getcwd())) == "src":
        sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
    from src.database.db import init_database, check_database_exists, get_db
    from src.database.users import (
        create_user, list_users, deactivate_user,
        reactivate_user, regenerate_api_key
    )
    from src.database.api_logs import (
        get_user_api_stats, get_system_api_stats,
        cleanup_old_logs, get_recent_errors
    )
    from src.database.presents import (
        get_present_classification_stats, cleanup_old_present_cache
    )


@click.group()
def cli():
    """Database management CLI for Predict Presents"""
    pass

@cli.command()
def init():
    """Initialize the database with schema"""
    if check_database_exists():
        if not click.confirm("Database already exists. Reinitialize? This will delete all data."):
            click.echo("Initialization cancelled.")
            return

    try:
        init_database()
        click.echo("✅ Database initialized successfully")
    except Exception as e:
        click.echo(f"❌ Error initializing database: {e}", err=True)
        sys.exit(1)

@cli.group()
def user():
    """User management commands"""
    pass

@user.command('create')
@click.option('--username', required=True, help='Username for new user')
def create_api_user(username: str):
    """Create a new API user"""
    try:
        result = create_user(username)
        click.echo(f"\n✅ User created successfully!")
        click.echo(f"Username: {result['username']}")
        click.echo(f"API Key: {result['api_key']}")
        click.echo("\n⚠️  Save this API key - it won't be shown again!")
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ An unexpected error occurred: {e}", err=True)
        sys.exit(1)


@user.command('list')
@click.option('--active-only', is_flag=True, help='Show only active users')
def list_api_users(active_only: bool):
    """List all API users"""
    try:
        users = list_users(active_only)
    except Exception as e:
        click.echo(f"❌ Error listing users: {e}", err=True)
        sys.exit(1)

    if not users:
        click.echo("No users found")
        return

    headers = ['ID', 'Username', 'Status', 'Created', 'Updated']
    rows = []

    for user_row in users:
        rows.append([
            user_row['id'],
            user_row['username'],
            "Active" if user_row['is_active'] else "Inactive",
            user_row['created_at'][:19] if user_row['created_at'] else "N/A",  # Trim microseconds
            user_row['updated_at'][:19] if user_row['updated_at'] else "N/A"
        ])

    click.echo(tabulate(rows, headers=headers, tablefmt='grid'))

@user.command('deactivate')
@click.option('--username', required=True, help='Username to deactivate')
def deactivate_api_user(username: str):
    """Deactivate a user"""
    try:
        if deactivate_user(username):
            click.echo(f"✅ User '{username}' deactivated")
        else:
            click.echo(f"❌ User '{username}' not found or already inactive", err=True)
    except Exception as e:
        click.echo(f"❌ Error deactivating user: {e}", err=True)
        sys.exit(1)


@user.command('reactivate')
@click.option('--username', required=True, help='Username to reactivate')
def reactivate_api_user(username: str):
    """Reactivate a user"""
    try:
        if reactivate_user(username):
            click.echo(f"✅ User '{username}' reactivated")
        else:
            click.echo(f"❌ User '{username}' not found or already active", err=True)
    except Exception as e:
        click.echo(f"❌ Error reactivating user: {e}", err=True)
        sys.exit(1)

@user.command('regenerate-key')
@click.option('--username', required=True, help='Username to regenerate key for')
def regenerate_key(username: str):
    """Generate a new API key for a user"""
    if not click.confirm(f"Generate new API key for '{username}'? This will invalidate the old key."):
        click.echo("Key regeneration cancelled.")
        return

    try:
        new_key = regenerate_api_key(username)
        if new_key:
            click.echo(f"\n✅ New API key generated!")
            click.echo(f"Username: {username}")
            click.echo(f"New API Key: {new_key}")
            click.echo("\n⚠️  Save this API key - it won't be shown again!")
        else:
            click.echo(f"❌ User '{username}' not found", err=True)
    except Exception as e:
        click.echo(f"❌ Error regenerating key: {e}", err=True)
        sys.exit(1)


@cli.group()
def stats():
    """View statistics and logs"""
    pass

@stats.command('user')
@click.option('--username', required=True, help='Username to get stats for')
@click.option('--days', default=30, type=int, help='Number of days to look back')
def user_stats_cmd(username: str, days: int): # Renamed to avoid conflict
    """Show API usage stats for a user"""
    try:
        # Get user ID
        all_users = list_users()
        user_data = next((u for u in all_users if u['username'] == username), None)

        if not user_data:
            click.echo(f"❌ User '{username}' not found", err=True)
            return

        user_id = user_data['id']
        stats_data = get_user_api_stats(user_id, days)

        click.echo(f"\n📊 API Usage Stats for {username} (last {days} days)")
        click.echo(f"Total Calls: {stats_data.get('total_calls', 0)}")
        click.echo(f"Error Count: {stats_data.get('error_count', 0)}")
        avg_resp_time = stats_data.get('avg_response_time')
        click.echo(f"Avg Response Time: {avg_resp_time:.2f}ms" if avg_resp_time is not None else "N/A")
        click.echo(f"Unique Endpoints: {stats_data.get('unique_endpoints', 0)}")

        if stats_data.get('endpoints'):
            click.echo("\nTop Endpoints:")
            headers = ['Endpoint', 'Calls', 'Avg Time (ms)']
            rows = []
            for e in stats_data['endpoints'][:5]:
                avg_time = e.get('avg_response_time')
                rows.append([
                    e['api_route'], 
                    e['call_count'], 
                    f"{avg_time:.2f}" if avg_time is not None else "N/A"
                ])
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            click.echo("No endpoint data available.")
    except Exception as e:
        click.echo(f"❌ Error fetching user stats: {e}", err=True)
        sys.exit(1)


@stats.command('system')
@click.option('--hours', default=24, type=int, help='Number of hours to look back')
def system_stats_cmd(hours: int): # Renamed to avoid conflict
    """Show system-wide API stats"""
    try:
        stats_data = get_system_api_stats(hours)

        click.echo(f"\n📊 System API Stats (last {hours} hours)")
        click.echo(f"Total Calls: {stats_data.get('total_calls', 0)}")
        click.echo(f"Active Users: {stats_data.get('active_users', 0)}")
        error_count = stats_data.get('error_count', 0)
        server_error_count = stats_data.get('server_error_count', 0)
        click.echo(f"Error Count: {error_count} ({server_error_count} server errors)")
        avg_resp_time = stats_data.get('avg_response_time')
        click.echo(f"Avg Response Time: {avg_resp_time:.2f}ms" if avg_resp_time is not None else "N/A")

        if stats_data.get('top_users'):
            click.echo("\nTop Users:")
            headers = ['Username', 'Calls']
            rows = [[u['username'], u['call_count']] for u in stats_data['top_users']]
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            click.echo("No top user data available.")
    except Exception as e:
        click.echo(f"❌ Error fetching system stats: {e}", err=True)
        sys.exit(1)


@stats.command('errors')
@click.option('--limit', default=10, type=int, help='Number of errors to show')
def recent_errors_cmd(limit: int): # Renamed to avoid conflict
    """Show recent API errors"""
    try:
        errors = get_recent_errors(limit)
    except Exception as e:
        click.echo(f"❌ Error fetching recent errors: {e}", err=True)
        sys.exit(1)

    if not errors:
        click.echo("No recent errors found")
        return

    click.echo(f"\n❌ Recent API Errors (showing {len(errors)})")

    for error in errors:
        click.echo(f"\n[{error.get('date_time', 'N/A')}] {error.get('username', 'N/A')} - {error.get('api_route', 'N/A')}")
        click.echo(f"Status: {error.get('response_status_code', 'N/A')}")
        if error.get('error_message'):
            click.echo(f"Error: {error['error_message']}")
        if error.get('request_payload'):
            click.echo(f"Request: {error['request_payload']}")


@stats.command('cache')
def cache_stats_cmd(): # Renamed to avoid conflict
    """Show product classification cache stats"""
    try:
        stats_data = get_present_classification_stats()
    except Exception as e:
        click.echo(f"❌ Error fetching cache stats: {e}", err=True)
        sys.exit(1)

    click.echo("\n📦 Present Classification Cache Stats")
    click.echo(f"Total Cached: {stats_data.get('total_cached', 0)}")
    click.echo(f"Unique Presents: {stats_data.get('unique_products', 0)}")
    click.echo(f"Successful: {stats_data.get('successful', 0)}")
    click.echo(f"Failed: {stats_data.get('failed', 0)}")

    if stats_data.get('oldest_entry'):
        click.echo(f"Oldest Entry: {stats_data['oldest_entry']}")
        click.echo(f"Newest Entry: {stats_data['newest_entry']}")

    if stats_data.get('top_categories'):
        click.echo("\nTop Categories:")
        headers = ['Category', 'Count']
        rows = [[c['item_main_category'], c['count']] for c in stats_data['top_categories']]
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
    else:
        click.echo("No category data available.")


@cli.group()
def cleanup():
    """Database cleanup commands"""
    pass

@cleanup.command('logs')
@click.option('--days', default=90, type=int, help='Keep logs from last N days')
def cleanup_logs_cmd(days: int): # Renamed to avoid conflict
    """Clean up old API logs"""
    if not click.confirm(f"Delete API logs older than {days} days? This cannot be undone."):
        click.echo("Log cleanup cancelled.")
        return

    try:
        deleted = cleanup_old_logs(days)
        click.echo(f"✅ Deleted {deleted} old log entries")
    except Exception as e:
        click.echo(f"❌ Error cleaning up logs: {e}", err=True)
        sys.exit(1)


@cleanup.command('cache')
@click.option('--days', default=30, type=int, help='Keep cache entries from last N days')
def cleanup_cache_cmd(days: int): # Renamed to avoid conflict
    """Clean up old product cache entries"""
    if not click.confirm(f"Delete cache entries older than {days} days? This cannot be undone."):
        click.echo("Cache cleanup cancelled.")
        return
    try:
        deleted = cleanup_old_present_cache(days)
        click.echo(f"✅ Deleted {deleted} old cache entries")
    except Exception as e:
        click.echo(f"❌ Error cleaning up cache: {e}", err=True)
        sys.exit(1)

@cli.command('reset-failed-presents')
@click.option('--status-filter', default='error%', help='SQL LIKE pattern for status to reset (default: error%)')
@click.option('--dry-run', is_flag=True, help='Show what would be reset without making changes')
def reset_failed_presents(status_filter: str, dry_run: bool):
    """Reset failed present classifications back to pending status."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # First, count how many would be affected
            cursor.execute("""
                SELECT COUNT(*) as count, classification_status,
                       COUNT(DISTINCT classification_status) as status_count
                FROM present_attributes
                WHERE classification_status LIKE ?
                GROUP BY classification_status
            """, (status_filter,))
            
            results = cursor.fetchall()
            
            if not results:
                click.echo("No failed presents found to reset.")
                return
                
            # Show what will be reset
            click.echo("\nPresents to reset:")
            total_count = 0
            for row in results:
                count = row['count']
                status = row['classification_status']
                click.echo(f"  - {count} presents with status '{status}'")
                total_count += count
            
            if dry_run:
                click.echo(f"\n🔍 DRY RUN: Would reset {total_count} presents to 'pending_classification'")
                return
                
            if not click.confirm(f"\nReset {total_count} presents to 'pending_classification'?"):
                click.echo("Reset cancelled.")
                return
                
            # Perform the reset
            cursor.execute("""
                UPDATE present_attributes
                SET classification_status = 'pending_classification',
                    thread_id = NULL,
                    run_id = NULL
                WHERE classification_status LIKE ?
            """, (status_filter,))
            
            affected = cursor.rowcount
            
            click.echo(f"✅ Successfully reset {affected} presents to 'pending_classification'")
        
    except Exception as e:
        click.echo(f"❌ Error resetting failed presents: {e}", err=True)
        sys.exit(1)

@cli.command('import-presents-csv')
@click.option('--csv-path', required=True, type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path), help='Path to the CSV file containing present attributes.')
def import_presents_csv(csv_path: Path):
    """Import present attributes from a CSV file into the present_attributes table."""
    click.echo(f"Attempting to import presents from: {csv_path}")

    try:
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=['']) # Treat empty strings as NA, but keep actual "NA" strings if any
    except FileNotFoundError:
        click.echo(f"❌ Error: CSV file not found at {csv_path}", err=True)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        click.echo(f"❌ Error: CSV file at {csv_path} is empty.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error reading CSV file: {e}", err=True)
        sys.exit(1)

    required_columns = {
        'present_hash', 'present_name', 'present_vendor', 'model_name', 'model_no',
        'item_main_category', 'item_sub_category', 'color', 'brand', 'vendor',
        'target_demographic', 'utility_type', 'durability', 'usage_type'
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        click.echo(f"❌ Error: CSV file is missing required columns: {', '.join(missing_columns)}", err=True)
        sys.exit(1)

    imported_count = 0
    skipped_count = 0
    error_count = 0

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for index, row in df.iterrows():
            present_hash = row['present_hash']
            try:
                cursor.execute("SELECT 1 FROM present_attributes WHERE present_hash = ?", (present_hash,))
                exists = cursor.fetchone()

                if exists:
                    click.echo(f"⏭️ Skipping existing present_hash: {present_hash}")
                    skipped_count += 1
                    continue

                # Prepare data for insertion
                # Convert pandas NA (often NaN for empty strings if not handled by keep_default_na/na_values) to None
                data_to_insert = {
                    "present_hash": present_hash,
                    "present_name": str(row['present_name']) if pd.notna(row['present_name']) else None,
                    "present_vendor": str(row['present_vendor']) if pd.notna(row['present_vendor']) else None,
                    "model_name": str(row['model_name']) if pd.notna(row['model_name']) else None,
                    "model_no": str(row['model_no']) if pd.notna(row['model_no']) else None,
                    "item_main_category": str(row['item_main_category']) if pd.notna(row['item_main_category']) else None,
                    "item_sub_category": str(row['item_sub_category']) if pd.notna(row['item_sub_category']) else None,
                    "color": str(row['color']) if pd.notna(row['color']) else None,
                    "brand": str(row['brand']) if pd.notna(row['brand']) else None,
                    "vendor": str(row['vendor']) if pd.notna(row['vendor']) else None, # Classified vendor
                    "value_price": None, # As per user request
                    "target_demographic": str(row['target_demographic']) if pd.notna(row['target_demographic']) else None,
                    "utility_type": str(row['utility_type']) if pd.notna(row['utility_type']) else None,
                    "durability": str(row['durability']) if pd.notna(row['durability']) else None,
                    "usage_type": str(row['usage_type']) if pd.notna(row['usage_type']) else None,
                    "classified_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "classification_status": "success", # Pre-classified
                    "thread_id": None,
                    "run_id": None,
                }
                
                # Ensure all required string fields that are not nullable in DB are not None
                # For simplicity, assuming all text fields in DB can be NULL or have defaults if not provided
                # Or that pandas `str(row['...']) if pd.notna(...) else None` handles it.

                columns = ', '.join(data_to_insert.keys())
                placeholders = ', '.join(['?'] * len(data_to_insert))
                sql = f"INSERT INTO present_attributes ({columns}) VALUES ({placeholders})"
                
                cursor.execute(sql, tuple(data_to_insert.values()))
                imported_count += 1
                if imported_count % 100 == 0: # Log progress
                    click.echo(f"Processed {index + 1} rows, Imported: {imported_count}, Skipped: {skipped_count}...")

            except Exception as e_row:
                click.echo(f"❌ Error processing row {index + 2} (present_hash: {present_hash}): {e_row}", err=True)
                error_count += 1
                # Optionally, decide if you want to rollback or continue on row error
        
        conn.commit()
        click.echo("\n--- Import Summary ---")
        click.echo(f"✅ Successfully imported: {imported_count} presents")
        click.echo(f"⏭️ Skipped (already exist): {skipped_count} presents")
        if error_count > 0:
            click.echo(f"❌ Errors encountered: {error_count} presents", err=True)
        click.echo("----------------------")

    except Exception as e:
        if conn:
            conn.rollback()
        click.echo(f"❌ An unexpected error occurred during import: {e}", err=True)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    # This allows running the CLI directly for development/testing
    # For production, it's better to install as a package or run as a module
    # e.g., python -m src.database.cli user list
    cli()