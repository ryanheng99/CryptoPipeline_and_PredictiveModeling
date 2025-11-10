from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from scripts.fetch_data import fetch_binance_price
from scripts.transform_data import transform_data
from scripts.store_data import store_data

DB_URL = "postgresql://user:password@postgres:5432/crypto"

def ingest_transform_store():
    df = fetch_binance_price()
    df = transform_data(df)
    store_data(df, DB_URL)

with DAG(
    "crypto_ingest_pipeline",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    schedule_interval="*/5 * * * *",
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task = PythonOperator(
        task_id="ingest_transform_store",
        python_callable=ingest_transform_store
    )
