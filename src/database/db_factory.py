"""
Database factory - PostgreSQL only implementation.
"""
import logging

# Direct import of PostgreSQL module
from . import db_postgres

logger = logging.getLogger(__name__)

# Direct delegation to PostgreSQL module
get_db = db_postgres.get_db
init_database = db_postgres.init_database
execute_query = db_postgres.execute_query
execute_write = db_postgres.execute_write
execute_many = db_postgres.execute_many
check_database_exists = db_postgres.check_database_exists

# Log that we're using PostgreSQL
logger.info("Database factory configured for PostgreSQL")