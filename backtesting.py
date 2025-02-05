import pandas as pd
import matplotlib.pyplot as plt

# Filepath to the CSV data
DATA_FILE = "data/BTC-USDT_5m.csv"  # Update this to the path of your data file


def load_data(filepath):
    """
    Load historical data from a CSV file.
    """
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Ensure numeric columns are floats
    numeric_cols = ["open", "high", "low", "close", "volume_base"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def simple_moving_average_strategy(df, fast_period=10, slow_period=20):
    """
    Simple moving average crossover strategy.
    Generates buy/sell signals based on SMA crossovers.
    """
    # Calculate moving averages
    df["SMA_Fast"] = df["close"].rolling(window=fast_period).mean()
    df["SMA_Slow"] = df["close"].rolling(window=slow_period).mean()

    # Generate signals
    df["Signal"] = 0
    df.loc[df["SMA_Fast"] > df["SMA_Slow"], "Signal"] = 1  # Buy signal
    df.loc[df["SMA_Fast"] < df["SMA_Slow"], "Signal"] = -1  # Sell signal

    return df


def backtest_strategy(df, initial_balance=1000, risk_per_trade=100):
    """
    Backtest the strategy on historical data.
    Tracks balance, PnL, and other metrics.
    """
    balance = initial_balance
    position = 0  # Current position (in units)
    trade_log = []  # Store trade details

    for i in range(1, len(df)):
        # Get the current signal
        signal = df["Signal"].iloc[i]

        # Get the current price
        price = df["close"].iloc[i]

        # Execute a trade if there's a signal
        if signal == 1 and position == 0:  # Buy signal
            # Calculate risk and position size
            risk = risk_per_trade * balance
            stop_loss = price * 0.99  # Example: 1% stop loss
            position_size = risk / (price - stop_loss)
            position = position_size
            balance -= position_size * price
            trade_log.append({"Type": "BUY", "Price": price, "Balance": balance})

        elif signal == -1 and position > 0:  # Sell signal
            # Close position
            balance += position * price
            trade_log.append({"Type": "SELL", "Price": price, "Balance": balance})
            position = 0

    # Convert trade log to DataFrame
    trade_log_df = pd.DataFrame(trade_log)
    return balance, trade_log_df


def visualize_results(df, trade_log):
    """
    Visualize backtest results, including signals and trade outcomes.
    """
    # Plot price and SMA
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["close"], label="Close Price", color="blue")
    plt.plot(df.index, df["SMA_Fast"], label="SMA Fast", color="orange")
    plt.plot(df.index, df["SMA_Slow"], label="SMA Slow", color="green")

    # Mark buy/sell signals
    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]
    plt.scatter(buys.index, buys["close"], marker="^", color="green", label="Buy Signal", alpha=1)
    plt.scatter(sells.index, sells["close"], marker="v", color="red", label="Sell Signal", alpha=1)

    # Add title and legend
    plt.title("Backtesting Results - Simple Moving Average Strategy", fontsize=16)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("Price (USDT)", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    # Plot trade log balance over time
    if not trade_log.empty:
        plt.figure(figsize=(14, 6))
        plt.plot(trade_log.index, trade_log["Balance"], label="Balance", color="purple")
        plt.title("Account Balance Over Time", fontsize=16)
        plt.xlabel("Trade #", fontsize=12)
        plt.ylabel("Balance (USDT)", fontsize=12)
        plt.grid()
        plt.legend()
        plt.show()


def main():
    """
    Main function to run the backtest.
    """
    # Load historical data
    print("Loading historical data...")
    df = load_data(DATA_FILE)

    # Apply strategy
    print("Applying moving average strategy...")
    df = simple_moving_average_strategy(df)

    # Backtest strategy
    print("Backtesting strategy...")
    final_balance, trade_log = backtest_strategy(df)

    # Display results
    print(f"Final Balance: ${final_balance:.2f}")
    print(trade_log)

    # Visualize results
    print("Visualizing results...")
    visualize_results(df, trade_log)


if __name__ == "__main__":
    main()
