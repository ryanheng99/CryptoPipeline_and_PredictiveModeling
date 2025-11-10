
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_with_arima(df):
    df = df.set_index("timestamp")
    df = df.resample("H").mean().dropna()
    model = ARIMA(df["price"], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=24)
    return forecast
