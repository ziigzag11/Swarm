import ccxt
import pandas as pd
import time
import os

BASE_DIR = r"C:\Users\erikn\Desktop\Trading Agents Swarm 3.10\Bot's"
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

exchange = ccxt.kraken({
    "rateLimit": 3000,
    "enableRateLimit": True,
})

def fetch_kraken_pairs():
    print("Fetching tradable pairs from Kraken...")
    markets = exchange.load_markets()
    usd_pairs = [pair for pair in markets if pair.endswith("/USD") and markets[pair]["active"]]
    print(f"Found {len(usd_pairs)} USD pairs: {usd_pairs}")
    return usd_pairs

def fetch_historical_data(pair, timeframe="5m", since=None, limit=1000):
    print(f"Fetching data for {pair}...")
    try:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Error fetching data for {pair}: {e}")
        return None

if __name__ == "__main__":
    pairs = fetch_kraken_pairs()
    for pair in pairs:
        df = fetch_historical_data(pair)
        if df is not None:
            filename = os.path.join(DATA_DIR, f"{pair.replace('/', '-')}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved data for {pair} to {filename}")
        time.sleep(exchange.rateLimit / 1000)
