import pandas as pd

def transform_data(df):
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.dropna(inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
