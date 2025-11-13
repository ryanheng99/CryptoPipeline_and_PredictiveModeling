import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform and clean cryptocurrency price data.
    
    Transformations:
    - Remove duplicates
    - Handle missing values
    - Calculate technical indicators
    - Add time-based features
    
    Args:
        df: Raw price DataFrame
        
    Returns:
        Transformed DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame received for transformation")
        return df
    
    df = df.copy()
    
    # 1. Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['symbol', 'timestamp'])
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate records")
    
    # 2. Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 3. Handle missing prices (forward fill, then backward fill)
    df['price'] = df['price'].fillna(method='ffill').fillna(method='bfill')
    
    # 4. Remove invalid prices
    df = df[df['price'] > 0]
    
    # 5. Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 6. Calculate technical indicators (if enough data)
    if len(df) >= 20:
        df = add_technical_indicators(df)
    
    logger.info(f"Transformed {len(df)} records")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators for analysis.
    
    Indicators:
    - Moving averages (SMA)
    - Rate of change (ROC)
    - Volatility
    - Price momentum
    
    Args:
        df: DataFrame with price column
        
    Returns:
        DataFrame with additional indicator columns
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
    df['sma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
    
    # Exponential Moving Average
    df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
    
    # Rate of Change
    df['roc_1h'] = df['price'].pct_change(periods=1) * 100
    df['roc_24h'] = df['price'].pct_change(periods=24) * 100
    
    # Volatility (rolling standard deviation)
    df['volatility_7d'] = df['price'].rolling(window=168, min_periods=1).std()
    
    # Price momentum
    df['momentum'] = df['price'] - df['price'].shift(12)
    
    # Bollinger Bands
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
    
    # RSI (Relative Strength Index)
    df['rsi'] = calculate_rsi(df['price'], periods=14)
    
    # Fill any NaN values created by rolling calculations
    df = df.fillna(method='bfill')
    
    logger.info("Added technical indicators")
    return df


def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Price series
        periods: RSI period (default 14)
        
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def normalize_prices(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize price data for ML models.
    
    Args:
        df: DataFrame with price column
        method: 'minmax' or 'zscore'
        
    Returns:
        DataFrame with normalized price column
    """
    df = df.copy()
    
    if method == 'minmax':
        # Min-Max scaling to [0, 1]
        min_price = df['price'].min()
        max_price = df['price'].max()
        df['price_normalized'] = (df['price'] - min_price) / (max_price - min_price)
        
    elif method == 'zscore':
        # Z-score normalization
        mean_price = df['price'].mean()
        std_price = df['price'].std()
        df['price_normalized'] = (df['price'] - mean_price) / std_price
    
    logger.info(f"Normalized prices using {method} method")
    return df


def create_sequences(df: pd.DataFrame, sequence_length: int = 24) -> tuple:
    """
    Create sequences for LSTM training.
    
    Args:
        df: DataFrame with price data
        sequence_length: Number of time steps per sequence
        
    Returns:
        Tuple of (X, y) arrays for training
    """
    prices = df['price'].values
    
    X, y = [], []
    for i in range(len(prices) - sequence_length):
        X.append(prices[i:i + sequence_length])
        y.append(prices[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created {len(X)} sequences of length {sequence_length}")
    return X, y


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute-level data to hourly OHLC.
    
    Args:
        df: DataFrame with timestamp and price columns
        
    Returns:
        Hourly aggregated DataFrame
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to hourly
    hourly = df.groupby(['symbol']).resample('1H').agg({
        'price': ['first', 'max', 'min', 'last', 'mean', 'count']
    })
    
    hourly.columns = ['open', 'high', 'low', 'close', 'avg', 'count']
    hourly = hourly.reset_index()
    
    logger.info(f"Aggregated to {len(hourly)} hourly records")
    return hourly


if __name__ == "__main__":
    # Test with sample data
    test_df = pd.DataFrame({
        'symbol': ['BTCUSDT'] * 100,
        'price': np.random.uniform(40000, 50000, 100),
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H')
    })
    
    print("Original data:")
    print(test_df.head())
    
    transformed = transform_data(test_df)
    print("\nTransformed data:")
    print(transformed.head())
    print(f"\nColumns: {transformed.columns.tolist()}")