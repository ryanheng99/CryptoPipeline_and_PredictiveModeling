import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def forecast_with_lstm(df):
    df = df.set_index("timestamp")
    df = df.resample("H").mean().dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df["price"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32)

    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    prediction = model.predict(last_60)
    return scaler.inverse_transform(prediction)
