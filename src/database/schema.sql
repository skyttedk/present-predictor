-- User table for API authentication
CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    api_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- API call logging
CREATE TABLE IF NOT EXISTS user_api_call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    api_route TEXT NOT NULL,
    request_payload TEXT,  -- JSON string
    response_payload TEXT, -- JSON string
    response_status_code INTEGER,
    response_time_ms REAL,
    error_message TEXT,
    FOREIGN KEY (user_id) REFERENCES user(id)
);

-- Present classification cache
DROP TABLE IF EXISTS present_attributes;
CREATE TABLE IF NOT EXISTS present_attributes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    present_hash TEXT NOT NULL UNIQUE,
    present_name TEXT,
    present_vendor TEXT,
    model_name TEXT,
    model_no TEXT,
    thread_id TEXT,
    run_id TEXT,
    item_main_category TEXT,
    item_sub_category TEXT,
    color TEXT,
    brand TEXT,
    vendor TEXT, -- This is the classified vendor attribute
    value_price REAL,
    target_demographic TEXT,
    utility_type TEXT,
    durability TEXT,
    usage_type TEXT,
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    classification_status TEXT DEFAULT 'success'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_api_key ON user(api_key);
CREATE INDEX IF NOT EXISTS idx_api_log_user_date ON user_api_call_log(user_id, date_time);
CREATE INDEX IF NOT EXISTS idx_api_log_route ON user_api_call_log(api_route);
CREATE INDEX IF NOT EXISTS idx_present_hash ON present_attributes(present_hash);

-- Trigger to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_user_timestamp
AFTER UPDATE ON user
BEGIN
    UPDATE user SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;