# Heroku Deployment Steps

## ✅ Completed Migration Tasks

### 1. Created Heroku Configuration Files
- ✅ `Procfile` - Configures gunicorn web server
- ✅ `runtime.txt` - Specifies Python 3.11.9
- ✅ Updated `requirements.txt` with production dependencies

### 2. Database Abstraction Layer
- ✅ `src/database/db_postgres.py` - PostgreSQL support
- ✅ `src/database/schema_postgres.sql` - PostgreSQL schema
- ✅ `src/database/db_factory.py` - Auto-switches between SQLite/PostgreSQL

### 3. Updated All Database Imports
- ✅ All modules now use `db_factory` instead of direct imports
- ✅ App automatically uses PostgreSQL when `DATABASE_URL` is set

### 4. Migration Script
- ✅ `scripts/migrate_to_postgres.py` - Transfers data from SQLite to PostgreSQL

## 📋 Next Steps for Deployment

### Step 1: Install Heroku CLI
```bash
# Windows (using Chocolatey)
choco install heroku

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Create Heroku App
```bash
# Login to Heroku
heroku login

# Create new app
heroku create predict-presents-api

# Add PostgreSQL addon
heroku addons:create heroku-postgresql:mini
```

### Step 3: Set Environment Variables
```bash
# Set production environment
heroku config:set ENVIRONMENT=production
heroku config:set DEBUG=false

# Set OpenAI credentials
heroku config:set OPENAI_API_KEY=your-openai-api-key
heroku config:set OPENAI_ASSISTANT_ID=asst_BuFvA6iXF4xSyQ4px7Q5zjiN

# Set security key
heroku config:set SECRET_KEY=$(openssl rand -hex 32)

# Verify settings
heroku config
```

### Step 4: Deploy to Heroku
```bash
# Add Heroku remote (if not already added)
heroku git:remote -a predict-presents-api

# Commit all changes
git add .
git commit -m "Prepare for Heroku deployment"

# Deploy to Heroku
git push heroku main
```

### Step 5: Initialize Database
```bash
# Run database initialization
heroku run python -c "from src.database import db_factory; db_factory.init_database()"
```

### Step 6: Migrate Data (Optional)
If you have existing data in SQLite:

```bash
# Get DATABASE_URL
heroku config:get DATABASE_URL

# Run migration locally
DATABASE_URL="postgresql://..." python scripts/migrate_to_postgres.py

# OR run on Heroku (upload SQLite file first)
heroku run python scripts/migrate_to_postgres.py
```

### Step 7: Create Initial User
```bash
# Connect to Heroku
heroku run python

# In Python shell:
from src.database.users import create_user
user = create_user("admin")
print(f"API Key: {user['api_key']}")
# SAVE THIS API KEY!
```

### Step 8: Test the API
```bash
# Get app URL
heroku info

# Test the endpoint
curl -H "X-API-Key: your-api-key" https://predict-presents-api.herokuapp.com/test
```

### Step 9: Monitor Logs
```bash
# View logs
heroku logs --tail

# View scheduler logs
heroku logs --tail --ps scheduler
```

## 🔧 Scheduler Configuration

The APScheduler is configured to run in-process. For production, consider:

### Option 1: Heroku Scheduler Addon (Recommended)
```bash
# Add scheduler addon
heroku addons:create scheduler:standard

# Open scheduler dashboard
heroku addons:open scheduler

# Add job: python -c "from src.scheduler.tasks import fetch_pending_present_attributes; import asyncio; asyncio.run(fetch_pending_present_attributes())"
# Frequency: Every 10 minutes
```

### Option 2: Worker Dyno (Costs Extra)
Create `worker.py`:
```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.scheduler.tasks import fetch_pending_present_attributes

scheduler = AsyncIOScheduler()
scheduler.add_job(fetch_pending_present_attributes, "interval", minutes=2)
scheduler.start()

# Keep running
asyncio.get_event_loop().run_forever()
```

Update `Procfile`:
```
web: gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}
worker: python worker.py
```

## 🚨 Important Notes

1. **Database**: The app will automatically use PostgreSQL when `DATABASE_URL` is set
2. **Dyno Sleep**: Free/Eco dynos sleep after 30 minutes of inactivity
3. **SSL**: Heroku provides SSL certificates automatically
4. **File Storage**: Don't store files on Heroku (ephemeral filesystem)

## 📊 Cost Breakdown
- **Eco Dyno**: $5/month (sleeps after 30 min)
- **Basic Dyno**: $7/month (never sleeps)
- **PostgreSQL Mini**: $5/month (10K rows)
- **Total**: ~$10-12/month minimum

## 🔍 Troubleshooting

### Database Connection Issues
```bash
# Check DATABASE_URL
heroku config:get DATABASE_URL

# Test connection
heroku run python -c "from src.database import db_factory; print(db_factory.check_database_exists())"
```

### Module Import Errors
```bash
# Ensure requirements are installed
heroku run pip list

# Check Python version
heroku run python --version
```

### Scheduler Not Running
- Check logs: `heroku logs --tail`
- Verify APScheduler is starting in main.py
- Consider using Heroku Scheduler addon instead

## ✅ Deployment Checklist
- [ ] Heroku CLI installed
- [ ] Heroku app created
- [ ] PostgreSQL addon added
- [ ] Environment variables set
- [ ] Code deployed
- [ ] Database initialized
- [ ] User created
- [ ] API tested
- [ ] Scheduler configured