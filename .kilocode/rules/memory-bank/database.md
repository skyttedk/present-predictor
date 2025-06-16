# Database Schema (SQLite)

This document outlines the SQLite database schema, as implemented in [`src/database/schema.sql`](src/database/schema.sql:1).

## Tables

### 1. `user`
Stores API user credentials and information.

**Columns**:
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `username` TEXT NOT NULL UNIQUE
- `api_key` TEXT NOT NULL UNIQUE (Hashed)
- `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- `is_active` BOOLEAN DEFAULT 1

**Trigger**:
- `update_user_timestamp`: Updates `updated_at` on any row modification.

### 2. `user_api_call_log`
Logs all API requests made by users.

**Columns**:
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `user_id` INTEGER NOT NULL (FOREIGN KEY REFERENCES `user(id)`)
- `date_time` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- `api_route` TEXT NOT NULL
- `request_payload` TEXT (JSON string)
- `response_payload` TEXT (JSON string)
- `response_status_code` INTEGER
- `response_time_ms` REAL
- `error_message` TEXT

### 3. `present_attributes`
Caches present classification results from OpenAI.

**Columns**:
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `present_hash` TEXT NOT NULL UNIQUE (MD5 hash of combined present_name, model_name, model_no; used for cache lookups)
- `present_name` TEXT
- `present_vendor` TEXT
- `model_name` TEXT
- `model_no` TEXT
- `thread_id` TEXT (OpenAI Assistant thread ID)
- `run_id` TEXT (OpenAI Assistant run ID)
- `item_main_category` TEXT
- `item_sub_category` TEXT
- `color` TEXT
- `brand` TEXT
- `vendor` TEXT
- `value_price` REAL
- `target_demographic` TEXT
- `utility_type` TEXT
- `durability` TEXT
- `usage_type` TEXT
- `classified_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- `classification_status` TEXT DEFAULT 'success' (e.g., 'success', 'error', 'pending')
- `vendor` TEXT (This is the classified vendor attribute)
- `value_price` REAL
- `target_demographic` TEXT
- `utility_type` TEXT
- `durability` TEXT
- `usage_type` TEXT
- `classified_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- `classification_status` TEXT DEFAULT 'success' (e.g., 'success', 'error', 'pending')

## Indexes
- `idx_user_api_key` ON `user(api_key)`
- `idx_api_log_user_date` ON `user_api_call_log(user_id, date_time)`
- `idx_api_log_route` ON `user_api_call_log(api_route)`
- `idx_present_hash` ON `present_attributes(present_hash)`

## Notes
- The `present_attributes` table uses `present_hash` (an MD5 hash of the combined present_name, model_name, and model_no) as the unique key for caching.
- OpenAI `thread_id` and `run_id` are stored for traceability and potential debugging of the classification process.
- The schema is defined in [`src/database/schema.sql`](src/database/schema.sql:1) and managed by modules in [`src/database/`](src/database/:1) (specifically [`src/database/presents.py`](src/database/presents.py:1)).
