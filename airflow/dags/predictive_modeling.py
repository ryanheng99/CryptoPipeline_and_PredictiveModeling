from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from scripts.fetch_data import fetch_binance_price
from scripts.transform_data import transform_data
from scripts.forecast_prices import forecast_with_prophet

def run_forecasting():
    df = fetch_binance_price()
    df = transform_data(df)
    forecast = forecast_with_prophet(df)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

with DAG(
    "crypto_forecasting",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    schedule_interval="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task = PythonOperator(
        task_id="forecast_prices",
        python_callable=run_forecasting
    )
