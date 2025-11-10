
## 📈 Crypto Market Data Pipeline
An end-to-end data engineering project that ingests, transforms, stores, detects anomalies, and forecasts real-time cryptocurrency prices using Binance API.


## 🚀 Features

Real-time price ingestion from Binance
Data transformation and cleaning
PostgreSQL storage
Anomaly detection using z-score
Forecasting with Prophet, ARIMA, and LSTM
Slack alerts for anomalies
Airflow orchestration
Dockerized setup


## 🧱 Tech Stack

Python
Airflow
PostgreSQL
Docker & Docker Compose
Prophet, ARIMA, LSTM
Slack Webhooks


## 📁 Project Structure
```
crypto_pipeline/
├── airflow/
│   └── dags/
│       ├── ingest_transform_store.py
│       ├── anomaly_detection.py
│       └── predictive_modeling.py
├── scripts/
│   ├── fetch_data.py
│   ├── transform_data.py
│   ├── store_data.py
│   ├── detect_anomalies.py
│   ├── forecast_prices.py
│   ├── arima_model.py
│   ├── lstm_model.py
│   └── slack_alert.py
├── docker-compose.yml
├── requirements.txt
├── .env
```



## 🛠️ Setup Instructions
# 1. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


# 2. Install Dependencies
pip install -r requirements.txt


# 3. Set Up Slack Alerts
Create a Slack Incoming Webhook and add it to .env:
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url


# 4. Start Docker Services
docker-compose up --build


# 5. Initialize Airflow
docker exec -it <airflow_container_name> bash
airflow db init
airflow users create \
    --username admin \
    --firstname Ryan \
    --lastname Heng \
    --role Admin \
    --email ryan@example.com \
    --password admin


# 6. Access Airflow UI
Go to http://localhost:8080 and log in.
# 7. Trigger DAGs
Enable and trigger:

crypto_ingest_pipeline
crypto_anomaly_detection
crypto_forecasting
# 8. Verify PostgreSQL Storage
psql -h localhost -U user -d crypto
SELECT * FROM crypto_prices LIMIT 10;


