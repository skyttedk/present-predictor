-- PostgreSQL schema for Predictive Gift Selection System (Safe initialization - no drops)

-- Create user table only if it doesn't exist
CREATE TABLE IF NOT EXISTS "user" (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    api_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE
);

-- Create trigger function for updating timestamp (only if it doesn't exist)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for user table only if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.triggers
        WHERE trigger_name = 'update_user_timestamp'
        AND event_object_table = 'user'
    ) THEN
        CREATE TRIGGER update_user_timestamp
            BEFORE UPDATE ON "user"
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- Create user_api_call_log table only if it doesn't exist
CREATE TABLE IF NOT EXISTS user_api_call_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id),
    date_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    api_route TEXT NOT NULL,
    request_payload TEXT,
    response_payload TEXT,
    response_status_code INTEGER,
    response_time_ms REAL,
    error_message TEXT
);

-- Create present_attributes table only if it doesn't exist
CREATE TABLE IF NOT EXISTS present_attributes (
    id SERIAL PRIMARY KEY,
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
    vendor TEXT,
    value_price REAL,
    target_demographic TEXT,
    utility_type TEXT,
    durability TEXT,
    usage_type TEXT,
    classified_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    classification_status TEXT DEFAULT 'success'
);

-- Create indexes only if they don't exist
CREATE INDEX IF NOT EXISTS idx_user_api_key ON "user"(api_key);
CREATE INDEX IF NOT EXISTS idx_api_log_user_date ON user_api_call_log(user_id, date_time);
CREATE INDEX IF NOT EXISTS idx_api_log_route ON user_api_call_log(api_route);
CREATE INDEX IF NOT EXISTS idx_present_hash ON present_attributes(present_hash);
CREATE INDEX IF NOT EXISTS idx_present_status ON present_attributes(classification_status);

-- Grant permissions (if needed for Heroku)
-- Note: Heroku automatically handles permissions for the database user