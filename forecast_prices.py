from prophet import Prophet

def forecast_with_prophet(df):
    df_prophet = df.rename(columns={"timestamp": "ds", "price": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    return forecast
