"""
CSV import functionality for present attributes.
Extracted from CLI for reuse in API endpoints.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
from pathlib import Path
import io

from .db_factory import get_db


def import_presents_from_csv(csv_content: str) -> Dict[str, Any]:
    """
    Import present attributes from CSV content.
    
    Args:
        csv_content: CSV content as string
        
    Returns:
        Dictionary with import statistics:
        {
            'imported_count': int,
            'skipped_count': int, 
            'error_count': int,
            'total_processed': int,
            'errors': List[str]  # List of error messages
        }
    """
    
    # Parse CSV content with enhanced error handling
    try:
        # First try with standard parsing
        df = pd.read_csv(
            io.StringIO(csv_content),
            keep_default_na=False,
            na_values=[''],
            encoding='utf-8'
        )
    except pd.errors.ParserError as e:
        # If standard parsing fails, try with more lenient options
        try:
            df = pd.read_csv(
                io.StringIO(csv_content),
                keep_default_na=False,
                na_values=[''],
                encoding='utf-8',
                quoting=1,  # QUOTE_ALL
                skipinitialspace=True,
                on_bad_lines='skip'  # Skip malformed lines
            )
            if df.empty:
                raise ValueError("CSV parsing resulted in empty dataframe after skipping bad lines")
        except Exception as e2:
            raise ValueError(f"Error parsing CSV content: {e}. Attempted lenient parsing also failed: {e2}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV content is empty")
    except Exception as e:
        raise ValueError(f"Error parsing CSV content: {e}")

    # Validate required columns
    required_columns = {
        'present_hash', 'present_name', 'present_vendor', 'model_name', 'model_no',
        'item_main_category', 'item_sub_category', 'color', 'brand', 'vendor',
        'target_demographic', 'utility_type', 'durability', 'usage_type'
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")

    imported_count = 0
    skipped_count = 0
    error_count = 0
    errors = []
    
    # Validate dataframe has data
    if df.empty:
        raise ValueError("CSV parsing resulted in empty dataframe")
    
    # Log parsing info for debugging
    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("No data rows found in CSV")

    try:
        # Use single database connection for the entire operation
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Step 1: Bulk check for existing hashes
            all_hashes = [row['present_hash'] for _, row in df.iterrows()]
            
            # Use IN clause for bulk existence check
            if all_hashes:
                placeholders = ','.join(['%s'] * len(all_hashes))
                check_query = f"SELECT present_hash FROM present_attributes WHERE present_hash IN ({placeholders})"
                cursor.execute(check_query, all_hashes)
                existing_hashes = {row['present_hash'] for row in cursor.fetchall()}
            else:
                existing_hashes = set()
            
            # Step 2: Prepare batch data for insertion
            batch_data = []
            current_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
            for index, row in df.iterrows():
                present_hash = row['present_hash']
                
                if present_hash in existing_hashes:
                    skipped_count += 1
                    continue
                
                try:
                    # Prepare data tuple for batch insert
                    insert_params = (
                        present_hash,
                        str(row['present_name']) if pd.notna(row['present_name']) else None,
                        str(row['present_vendor']) if pd.notna(row['present_vendor']) else None,
                        str(row['model_name']) if pd.notna(row['model_name']) else None,
                        str(row['model_no']) if pd.notna(row['model_no']) else None,
                        str(row['item_main_category']) if pd.notna(row['item_main_category']) else None,
                        str(row['item_sub_category']) if pd.notna(row['item_sub_category']) else None,
                        str(row['color']) if pd.notna(row['color']) else None,
                        str(row['brand']) if pd.notna(row['brand']) else None,
                        str(row['vendor']) if pd.notna(row['vendor']) else None,
                        None,  # value_price
                        str(row['target_demographic']) if pd.notna(row['target_demographic']) else None,
                        str(row['utility_type']) if pd.notna(row['utility_type']) else None,
                        str(row['durability']) if pd.notna(row['durability']) else None,
                        str(row['usage_type']) if pd.notna(row['usage_type']) else None,
                        current_timestamp,
                        'success',  # classification_status
                        None,  # thread_id
                        None   # run_id
                    )
                    batch_data.append(insert_params)
                    
                except Exception as e_row:
                    error_msg = f"Row {index + 2} (present_hash: {present_hash}): {e_row}"
                    errors.append(error_msg)
                    error_count += 1
            
            # Step 3: Bulk insert if we have data
            if batch_data:
                insert_query = """
                    INSERT INTO present_attributes (
                        present_hash, present_name, present_vendor, model_name, model_no,
                        item_main_category, item_sub_category, color, brand, vendor,
                        value_price, target_demographic, utility_type, durability, usage_type,
                        classified_at, classification_status, thread_id, run_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """
                
                # Use executemany for batch insert
                cursor.executemany(insert_query, batch_data)
                imported_count = len(batch_data)
            
            # Commit the transaction
            conn.commit()

    except Exception as e:
        raise ValueError(f"Database error during import: {e}")

    total_processed = imported_count + skipped_count + error_count

    return {
        'imported_count': imported_count,
        'skipped_count': skipped_count,
        'error_count': error_count,
        'total_processed': total_processed,
        'errors': errors
    }


def import_presents_from_file(csv_path: Path) -> Dict[str, Any]:
    """
    Import present attributes from CSV file path.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary with import statistics
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_content = f.read()
        return import_presents_from_csv(csv_content)
    except FileNotFoundError:
        raise ValueError(f"CSV file not found at {csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")