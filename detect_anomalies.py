import pandas as pd

def detect_anomalies(df):
    df["zscore"] = (df["price"] - df["price"].mean()) / df["price"].std()
    anomalies = df[df["zscore"].abs() > 3]
    return anomalies
