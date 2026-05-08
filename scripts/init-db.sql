-- =============================================================================
-- AI Vision Platform — PostgreSQL Initialisation Script
-- Runs once on first container start (docker-entrypoint-initdb.d).
-- The database itself is created automatically by the Postgres Docker image
-- using the POSTGRES_DB environment variable.
-- =============================================================================

-- Enable the uuid-ossp extension for UUID primary keys (used by SQLAlchemy models)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Application tables are created by SQLAlchemy on backend startup (init_db()).
-- This script is intentionally minimal — schema migrations are handled in code.
