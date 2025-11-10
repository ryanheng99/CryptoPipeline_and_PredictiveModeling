import os
import requests

def send_slack_alert(message):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("Slack webhook URL not set.")
        return

    payload = {"text": message}
    response = requests.post(webhook_url, json=payload)
    if response.status_code != 200:
        print(f"Failed to send alert: {response.text}")
    else:
        print("Slack alert sent.")
