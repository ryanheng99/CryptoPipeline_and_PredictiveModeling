
 datetime import datetime

def fetch_binance_price(
    symbol: str = "BTCUSDT",
    max_retries: int = 3,
    retry_delay: int = 2
) -> pd.DataFrame:
    """
    Fetch current price from Binance API with retry logic.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        DataFrame with columns: symbol, price, timestamp
        
    Raises:
        BinanceAPIError: If all retry attempts fail
    """
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching price for {symbol} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response
            if "symbol" not in data or "price" not in data:
                raise BinanceAPIError(f"Invalid response format: {data}")
            
            df = pd.DataFrame([{
                "symbol": data["symbol"],
                "price": float(data["price"]),
                "timestamp": datetime.utcnow()
            }])
            
            logger.info(f"Successfully fetched {symbol}: ${data['price']}")
            return df
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                logger.warning(f"Rate limited. Waiting {retry_delay * 2} seconds...")
                time.sleep(retry_delay * 2)
            else:
                logger.error(f"HTTP error: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt + 1}: {e}")
            
        except (ValueError, KeyError) as e:
            logger.error(f"Data parsing error: {e}")
            raise BinanceAPIError(f"Failed to parse response: {e}")
        
        # Wait before retry (except on last attempt)
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    # All retries failed
    error_msg = f"Failed to fetch data for {symbol} after {max_retries} attempts"
    logger.error(error_msg)
    raise BinanceAPIError(error_msg)


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 100
) -> pd.DataFrame:
    """
    Fetch historical candlestick data for model training.
    
    Args:
        symbol: Trading pair symbol
        interval: Kline interval (1m, 5m, 1h, 1d, etc.)
        limit: Number of data points (max 1000)
        
    Returns:
        DataFrame with OHLCV data
    """
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        df['symbol'] = symbol
        
        logger.info(f"Fetched {len(df)} historical records for {symbol}")
        return df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        logger.error(f"Error fetching klines: {e}")
        raise BinanceAPIError(f"Failed to fetch klines: {e}")


def fetch_multiple_symbols(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch prices for multiple symbols.
    
    Args:
        symbols: List of trading pair symbols
        
    Returns:
        Combined DataFrame with all symbols
    """
    all_data = []
    
    for symbol in symbols:
        try:
            df = fetch_binance_price(symbol)
            all_data.append(df)
            time.sleep(0.1)  # Rate limiting
        except BinanceAPIError as e:
            logger.error(f"Skipping {symbol}: {e}")
            continue
    
    if not all_data:
        raise BinanceAPIError("Failed to fetch data for any symbol")
    
    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    # Test single symbol
    df = fetch_binance_price()
    print("Current Price:")
    print(df)
    
    # Test historical data
    print("\nHistorical Data:")
    hist_df = fetch_binance_klines(limit=10)
    print(hist_df)
    
    # Test multiple symbols
    print("\nMultiple Symbols:")
    multi_df = fetch_multiple_symbols(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    print(multi_df)