import os
import requests
import pandas as pd
import time
import logging
from dotenv import load_dotenv

BASE_DIR = r"C:\Users\erikn\Desktop\Trading Agents Swarm 3.10\Bot's"
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("BL0FIN_API_KEY")
SECRET_KEY = os.getenv("BL0FIN_SECRET_KEY")
BASE_URL = "https://openapi.blofin.com"

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "trading_bot.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

trade_stats = {
    "total_profit_loss": 0.0,
    "total_trades": 0,
    "winning_trades": 0,
}

def fetch_market_data(symbol, timeframe="5m", limit=50):
    endpoint = f"{BASE_URL}/api/v1/market/candles"
    headers = {"X-ACCESS-KEY": API_KEY}
    params = {"instId": symbol, "interval": timeframe, "limit": limit}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json().get("data", [])
            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            df = pd.DataFrame([d[:6] for d in data], columns=columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        logging.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Data fetch error: {e}")
    return None

# [Keep other functions identical but ensure all file paths use BASE_DIR]

def run_bot():
    symbol = "BTC-USDT"
    timeframe = "5m"
    starting_capital = 10000
    risk_per_trade = 0.02 * starting_capital
    leverage = 10

    try:
        while True:
            df = fetch_market_data(symbol, timeframe)
            if df is None:
                time.sleep(10)
                continue

            trade_details = simple_strategy(df)
            
            if isinstance(trade_details, str):
                logging.info("No trade signal. Holding...")
                time.sleep(300)
                continue

            if trade_details["action"] == "buy":
                place_buy_order(
                    symbol=symbol,
                    entry_price=trade_details["entry_price"],
                    stop_loss_price=trade_details["stop_loss_price"],
                    take_profit_price=trade_details["take_profit_price"],
                    risk_amount=risk_per_trade,
                    leverage=leverage,
                )
            elif trade_details["action"] == "sell":
                place_sell_order(
                    symbol=symbol,
                    entry_price=trade_details["entry_price"],
                    stop_loss_price=trade_details["stop_loss_price"],
                    take_profit_price=trade_details["take_profit_price"],
                    risk_amount=risk_per_trade,
                    leverage=leverage,
                )

            time.sleep(300)
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")

if __name__ == "__main__":
    run_bot()