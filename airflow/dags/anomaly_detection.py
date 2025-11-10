
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from scripts.fetch_data import fetch_binance_price
from scripts.transform_data import transform_data
from scripts.detect_anomalies import detect_anomalies
from scripts.slack_alert import send_slack_alert

def run_anomaly_detection():
    df = fetch_binance_price()
    df = transform_data(df)
    anomalies = detect_anomalies(df)
    if not anomalies.empty:
        message = f"Anomalies detected: {anomalies.to_dict()}"
        send_slack_alert(message)

with DAG(
    "crypto_anomaly_detection",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    schedule_interval="@hourly",
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task = PythonOperator(
        task_id="detect_anomalies",
        python_callable=run_anomaly_detection
    )
