import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect anomalies in cryptocurrency price data.
    
    Methods:
    - zscore: Statistical z-score method
    - iqr: Interquartile range method
    - isolation_forest: ML-based isolation forest
    
    Args:
        df: DataFrame with price and timestamp columns
        method: Detection method
        threshold: Anomaly threshold (for zscore and iqr)
        
    Returns:
        DataFrame containing only anomalous records
    """
    if df.empty or len(df) < 10:
        logger.warning("Insufficient data for anomaly detection")
        return pd.DataFrame()
    
    df = df.copy()
    
    if method == 'zscore':
        anomalies = detect_zscore_anomalies(df, threshold)
    elif method == 'iqr':
        anomalies = detect_iqr_anomalies(df, threshold)
    elif method == 'isolation_forest':
        anomalies = detect_isolation_forest_anomalies(df)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Detected {len(anomalies)} anomalies using {method} method")
    return anomalies


def detect_zscore_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies using z-score method.
    
    Anomaly: |z-score| > threshold
    
    Args:
        df: DataFrame with price column
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        DataFrame with anomalies
    """
    df = df.copy()
    
    # Calculate z-scores
    mean = df['price'].mean()
    std = df['price'].std()
    
    if std == 0:
        logger.warning("Zero standard deviation, cannot calculate z-scores")
        return pd.DataFrame()
    
    df['z_score'] = (df['price'] - mean) / std
    
    # Identify anomalies
    df['is_anomaly'] = np.abs(df['z_score']) > threshold
    
    anomalies = df[df['is_anomaly']].copy()
    anomalies = anomalies.drop(columns=['is_anomaly'])
    
    return anomalies


def detect_iqr_anomalies(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect anomalies using Interquartile Range (IQR) method.
    
    Anomaly: value < Q1 - multiplier*IQR or value > Q3 + multiplier*IQR
    
    Args:
        df: DataFrame with price column
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        DataFrame with anomalies
    """
    df = df.copy()
    
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    df['is_anomaly'] = (df['price'] < lower_bound) | (df['price'] > upper_bound)
    df['iqr_lower'] = lower_bound
    df['iqr_upper'] = upper_bound
    
    anomalies = df[df['is_anomaly']].copy()
    anomalies = anomalies.drop(columns=['is_anomaly'])
    
    return anomalies


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.1
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest algorithm.
    
    Args:
        df: DataFrame with price and technical indicators
        contamination: Expected proportion of outliers (default 0.1)
        
    Returns:
        DataFrame with anomalies
    """
    df = df.copy()
    
    # Select features for anomaly detection
    feature_cols = ['price']
    
    # Add technical indicators if available
    optional_features = ['roc_1h', 'roc_24h', 'volatility_7d', 'rsi']
    for col in optional_features:
        if col in df.columns:
            feature_cols.append(col)
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    predictions = iso_forest.fit_predict(X_scaled)
    
    # -1 indicates anomaly, 1 indicates normal
    df['is_anomaly'] = predictions == -1
    df['anomaly_score'] = iso_forest.score_samples(X_scaled)
    
    anomalies = df[df['is_anomaly']].copy()
    anomalies = anomalies.drop(columns=['is_anomaly'])
    
    return anomalies


def detect_moving_average_anomalies(
    df: pd.DataFrame,
    window: int = 24,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Detect anomalies based on deviation from moving average.
    
    Anomaly: |price - MA| / MA > threshold
    
    Args:
        df: DataFrame with price column
        window: Moving average window
        threshold: Percentage deviation threshold
        
    Returns:
        DataFrame with anomalies
    """
    df = df.copy()
    
    df['ma'] = df['price'].rolling(window=window, min_periods=1).mean()
    df['deviation'] = np.abs(df['price'] - df['ma']) / df['ma']
    
    df['is_anomaly'] = df['deviation'] > threshold
    
    anomalies = df[df['is_anomaly']].copy()
    anomalies = anomalies.drop(columns=['is_anomaly', 'ma'])
    
    return anomalies


def detect_price_spike_anomalies(
    df: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Detect sudden price spikes (rapid changes).
    
    Anomaly: |price change| > threshold * price
    
    Args:
        df: DataFrame with price column
        threshold: Percentage change threshold
        
    Returns:
        DataFrame with spike anomalies
    """
    df = df.copy()
    
    df['price_change'] = df['price'].pct_change()
    df['is_spike'] = np.abs(df['price_change']) > threshold
    
    spikes = df[df['is_spike']].copy()
    spikes = spikes.drop(columns=['is_spike'])
    
    return spikes


def detect_multi_method_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple anomaly detection methods for robust detection.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with anomalies detected by multiple methods
    """
    df = df.copy()
    
    # Run multiple methods
    zscore_anomalies = detect_zscore_anomalies(df, threshold=3.0)
    iqr_anomalies = detect_iqr_anomalies(df, multiplier=1.5)
    
    # Combine results
    zscore_indices = set(zscore_anomalies.index)
    iqr_indices = set(iqr_anomalies.index)
    
    # Keep anomalies detected by at least 2 methods
    combined_indices = zscore_indices.intersection(iqr_indices)
    
    anomalies = df.loc[list(combined_indices)].copy()
    
    logger.info(f"Multi-method detection: {len(anomalies)} confirmed anomalies")
    return anomalies


def get_anomaly_summary(anomalies: pd.DataFrame) -> dict:
    """
    Generate summary statistics for detected anomalies.
    
    Args:
        anomalies: DataFrame with anomaly records
        
    Returns:
        Dictionary with summary statistics
    """
    if anomalies.empty:
        return {
            'count': 0,
            'mean_price': None,
            'max_price': None,
            'min_price': None
        }
    
    summary = {
        'count': len(anomalies),
        'mean_price': anomalies['price'].mean(),
        'max_price': anomalies['price'].max(),
        'min_price': anomalies['price'].min(),
        'mean_z_score': anomalies['z_score'].mean() if 'z_score' in anomalies else None,
        'latest_timestamp': anomalies['timestamp'].max()
    }
    
    return summary


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Generate normal data
    normal_prices = np.random.normal(50000, 1000, 100)
    
    # Add some anomalies
    anomaly_prices = [60000, 40000, 65000]
    prices = np.concatenate([normal_prices, anomaly_prices])
    
    test_df = pd.DataFrame({
        'symbol': ['BTCUSDT'] * len(prices),
        'price': prices,
        'timestamp': pd.date_range(start='2024-01-01', periods=len(prices), freq='H')
    })
    
    print("Testing anomaly detection methods:\n")
    
    # Test z-score method
    anomalies_zscore = detect_anomalies(test_df, method='zscore', threshold=2.5)
    print(f"Z-score method: {len(anomalies_zscore)} anomalies")
    print(anomalies_zscore[['price', 'z_score']].tail())
    
    # Test IQR method
    anomalies_iqr = detect_anomalies(test_df, method='iqr')
    print(f"\nIQR method: {len(anomalies_iqr)} anomalies")
    
    # Summary
    summary = get_anomaly_summary(anomalies_zscore)
    print(f"\nAnomaly Summary: {summary}")