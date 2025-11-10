from datetime import datetime

def fetch_binance_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame([{
            "symbol": data["symbol"],
            "price": float(data["price"]),
            "timestamp": datetime.utcnow()
        }])
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_binance_price()
    print(df)
