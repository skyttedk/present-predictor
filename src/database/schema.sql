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

-- Product classification cache
CREATE TABLE IF NOT EXISTS product_attributes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL UNIQUE,
    product_hash TEXT NOT NULL UNIQUE,
    thread_id TEXT,
    run_id TEXT,
    item_main_category TEXT,
    item_sub_category TEXT,
    color TEXT,
    brand TEXT,
    vendor TEXT,
    value_price REAL,
    target_demographic TEXT,
    utility_type TEXT,
    durability TEXT,
    usage_type TEXT,
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    classification_status TEXT DEFAULT 'success',
    raw_description TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_api_key ON user(api_key);
CREATE INDEX IF NOT EXISTS idx_api_log_user_date ON user_api_call_log(user_id, date_time);
CREATE INDEX IF NOT EXISTS idx_api_log_route ON user_api_call_log(api_route);
CREATE INDEX IF NOT EXISTS idx_product_hash ON product_attributes(product_hash);
CREATE INDEX IF NOT EXISTS idx_product_id ON product_attributes(product_id);

-- Trigger to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_user_timestamp
AFTER UPDATE ON user
BEGIN
    UPDATE user SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;