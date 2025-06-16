# Heroku Deployment Guide

## Current Issues & Required Changes

### 🔴 Critical Blockers

#### 1. **Database Migration (SQLite → PostgreSQL)**
Current: The app uses SQLite with local file storage (`predict_presents.db`)
Issue: Heroku's ephemeral filesystem wipes data on dyno restart

**Solution:**
```python
# src/database/db_postgres.py (NEW FILE)
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from urllib.parse import urlparse

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

@contextmanager
def get_db():
    """PostgreSQL connection for Heroku."""
    parsed = urlparse(DATABASE_URL)
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port,
        user=parsed.username,
        password=parsed.password,
        database=parsed.path[1:],
        cursor_factory=RealDictCursor
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

#### 2. **Background Scheduler Issues**
Current: APScheduler runs in-process
Issue: Heroku free/eco dynos sleep after 30 minutes

**Solutions:**
- **Option A**: Use Heroku Scheduler addon (recommended for simple tasks)
- **Option B**: Dedicated worker dyno with `worker: python src/scheduler/worker.py`
- **Option C**: Use Redis + Celery for robust task management

#### 3. **Environment Configuration**
Required changes to `.env` variables:
```bash
# Heroku assigns PORT dynamically
PORT=$PORT  # Don't hardcode!

# Database URL from Heroku Postgres addon
DATABASE_URL=postgresql://...

# Disable debug in production
DEBUG=false
ENVIRONMENT=production

# Add database path override
DATABASE_PATH=:memory:  # For SQLite fallback during migration
```

### ✅ Files Created
1. `Procfile` - Web server configuration
2. `runtime.txt` - Python 3.11.9 specification
3. Updated `requirements.txt` with gunicorn

### 📋 Deployment Steps

1. **Install Heroku CLI**
   ```bash
   # Windows
   choco install heroku
   # macOS
   brew install heroku/brew/heroku
   ```

2. **Create Heroku App**
   ```bash
   heroku create predict-presents-api
   heroku addons:create heroku-postgresql:mini
   ```

3. **Configure Environment Variables**
   ```bash
   heroku config:set ENVIRONMENT=production
   heroku config:set DEBUG=false
   heroku config:set OPENAI_API_KEY=your-key-here
   heroku config:set OPENAI_ASSISTANT_ID=asst_BuFvA6iXF4xSyQ4px7Q5zjiN
   heroku config:set SECRET_KEY=generate-secure-key-here
   ```

4. **Database Migration Script**
   ```python
   # scripts/migrate_to_postgres.py
   import sqlite3
   import psycopg2
   import os
   from urllib.parse import urlparse

   def migrate_data():
       # Connect to SQLite
       sqlite_conn = sqlite3.connect('predict_presents.db')
       sqlite_conn.row_factory = sqlite3.Row
       
       # Connect to PostgreSQL
       DATABASE_URL = os.getenv("DATABASE_URL")
       if DATABASE_URL.startswith("postgres://"):
           DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
       
       parsed = urlparse(DATABASE_URL)
       pg_conn = psycopg2.connect(
           host=parsed.hostname,
           port=parsed.port,
           user=parsed.username,
           password=parsed.password,
           database=parsed.path[1:]
       )
       
       # Create schema in PostgreSQL
       with open('src/database/schema_postgres.sql', 'r') as f:
           pg_conn.cursor().execute(f.read())
       pg_conn.commit()
       
       # Migrate data table by table
       tables = ['user', 'user_api_call_log', 'present_attributes']
       for table in tables:
           # ... migration logic
   ```

5. **Deploy to Heroku**
   ```bash
   git add .
   git commit -m "Prepare for Heroku deployment"
   git push heroku main
   ```

6. **Run Database Migration**
   ```bash
   heroku run python scripts/migrate_to_postgres.py
   ```

### 🔧 Code Modifications Needed

1. **Update src/config/settings.py**
   ```python
   # Add Heroku-specific settings
   port: int = Field(default=lambda: int(os.getenv("PORT", 8000)), env="PORT")
   database_url: str = Field(default=None, env="DATABASE_URL")
   ```

2. **Create Database Abstraction Layer**
   ```python
   # src/database/db_factory.py
   import os
   
   def get_db_module():
       if os.getenv("DATABASE_URL"):
           from . import db_postgres as db_module
       else:
           from . import db as db_module
       return db_module
   ```

3. **PostgreSQL Schema Conversion**
   - Convert SQLite `AUTOINCREMENT` to PostgreSQL `SERIAL`
   - Change `TIMESTAMP` to `TIMESTAMPTZ`
   - Update trigger syntax for PostgreSQL

### ⚠️ Additional Considerations

1. **File Storage**: Any uploaded files need external storage (S3, Cloudinary)
2. **Logging**: Use Heroku's logging with `heroku logs --tail`
3. **Monitoring**: Add New Relic or similar APM
4. **SSL**: Heroku provides SSL certificates automatically
5. **Scaling**: Start with 1 web dyno, scale as needed

### 🚀 Quick Start (Development to Heroku)

```bash
# 1. Create Heroku app
heroku create your-app-name

# 2. Add PostgreSQL
heroku addons:create heroku-postgresql:mini

# 3. Set environment variables
heroku config:set ENVIRONMENT=production DEBUG=false
heroku config:set OPENAI_API_KEY=your-key-here

# 4. Deploy
git push heroku main

# 5. Check logs
heroku logs --tail
```

### 📊 Cost Estimates (Monthly)
- **Eco Dyno**: $5/month (sleeps after 30 min inactivity)
- **Basic Dyno**: $7/month (never sleeps)
- **PostgreSQL Mini**: $5/month (10K rows)
- **Scheduler Addon**: Free
- **Total**: ~$10-12/month for basic setup

### 🎯 Recommended Architecture for Production

```
┌─────────────────┐     ┌──────────────────┐
│   Web Dyno      │────▶│ PostgreSQL DB    │
│  (FastAPI)      │     │ (Heroku Addon)   │
└─────────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Scheduler Addon │────▶│  Redis Cache     │
│ (Every 2 min)   │     │ (Redis To Go)    │
└─────────────────┘     └──────────────────┘
```

### 🛑 DO NOT Deploy Until:
1. ✅ PostgreSQL migration is complete
2. ✅ Environment variables are configured
3. ✅ Scheduler solution is chosen
4. ✅ Local testing with PostgreSQL passes
5. ✅ API authentication is properly configured