from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from fetch_data import fetch_binance_price, BinanceAPIError
from transform_data import transform_data
from detect_anomalies import detect_anomalies
from slack_alert import send_slack_alert
from store_data import store_data, store_anomalies, get_historical_data
import logging

logger = logging.getLogger(__name__)


def fetch_and_store_task(**context):
    """Fetch current price and store to database."""
    try:
        # Fetch data
        df = fetch_binance_price(symbol="BTCUSDT")
        
        if df.empty:
            raise ValueError("Received empty dataframe from API")
        
        # Store raw data
        rows_stored = store_data(df, table_name="crypto_prices")
        
        # Push to XCom for next task
        context['task_instance'].xcom_push(key='price_data', value=df.to_dict('records'))
        
        logger.info(f"Stored {rows_stored} price records")
        return rows_stored
        
    except BinanceAPIError as e:
        logger.error(f"API error: {e}")
        send_slack_alert(f"⚠️ API Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        send_slack_alert(f"🔥 Critical Error in fetch_and_store: {str(e)}")
        raise


def detect_anomalies_task(**context):
    """Detect anomalies in recent price data."""
    try:
        # Get historical data for anomaly detection
        df = get_historical_data(symbol="BTCUSDT", hours=168)  # 7 days
        
        if df.empty or len(df) < 10:
            logger.warning("Insufficient historical data for anomaly detection")
            return {"anomalies_found": 0}
        
        # Transform data
        df_transformed = transform_data(df)
        
        # Detect anomalies
        anomalies = detect_anomalies(df_transformed, threshold=3.0)
        
        if not anomalies.empty:
            # Store anomalies
            anomalies['detected_at'] = datetime.utcnow()
            store_anomalies(anomalies)
            
            # Send alert
            message = (
                f"🚨 *Anomaly Alert*\n"
                f"Symbol: {anomalies.iloc[0]['symbol']}\n"
                f"Price: ${anomalies.iloc[0]['price']:.2f}\n"
                f"Z-Score: {anomalies.iloc[0]['z_score']:.2f}\n"
                f"Detected: {len(anomalies)} anomalies"
            )
            send_slack_alert(message)
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            return {"anomalies_found": len(anomalies)}
        else:
            logger.info("No anomalies detected")
            return {"anomalies_found": 0}
            
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        send_slack_alert(f"❌ Anomaly Detection Failed: {str(e)}")
        raise


def send_summary_task(**context):
    """Send daily summary of anomaly detection."""
    ti = context['task_instance']
    anomaly_result = ti.xcom_pull(task_ids='detect_anomalies')
    
    if anomaly_result:
        message = (
            f"📊 *Daily Anomaly Detection Summary*\n"
            f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n"
            f"Anomalies Found: {anomaly_result.get('anomalies_found', 0)}\n"
            f"Status: {'⚠️ Attention Required' if anomaly_result['anomalies_found'] > 0 else '✅ Normal'}"
        )
        send_slack_alert(message)


# Default args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30)
}

# Define the DAG
with DAG(
    dag_id='crypto_anomaly_detection',
    default_args=default_args,
    description='Detect anomalies in cryptocurrency prices',
    schedule_interval='@hourly',  # Run every hour
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'anomaly', 'monitoring']
) as dag:
    
    # Task 1: Fetch and store price data
    fetch_task = PythonOperator(
        task_id='fetch_and_store',
        python_callable=fetch_and_store_task,
        provide_context=True
    )
    
    # Task 2: Detect anomalies
    detect_task = PythonOperator(
        task_id='detect_anomalies',
        python_callable=detect_anomalies_task,
        provide_context=True
    )
    
    # Task 3: Send summary (runs daily at 9 AM)
    summary_task = PythonOperator(
        task_id='send_summary',
        python_callable=send_summary_task,
        provide_context=True,
        trigger_rule='all_done'  # Run even if previous tasks fail
    )
    
    # Define task dependencies
    fetch_task >> detect_task >> summary_task