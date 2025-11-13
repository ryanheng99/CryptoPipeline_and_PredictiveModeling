#!/bin/bash
set -e

# Create crypto database and user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE crypto;
    CREATE USER user WITH PASSWORD 'password';
    GRANT ALL PRIVILEGES ON DATABASE crypto TO user;
EOSQL

# Connect to crypto database and create tables
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "crypto" <<-EOSQL
    CREATE TABLE IF NOT EXISTS crypto_prices (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        price DECIMAL(20, 8) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_symbol_timestamp (symbol, timestamp)
    );

    CREATE TABLE IF NOT EXISTS anomalies (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        price DECIMAL(20, 8) NOT NULL,
        z_score DECIMAL(10, 4),
        detected_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS forecasts (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        model_type VARCHAR(20) NOT NULL,
        forecast_date TIMESTAMP NOT NULL,
        predicted_price DECIMAL(20, 8) NOT NULL,
        confidence_lower DECIMAL(20, 8),
        confidence_upper DECIMAL(20, 8),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO user;
EOSQL

echo "Database initialization completed!"