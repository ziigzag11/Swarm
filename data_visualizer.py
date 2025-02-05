import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Filepath to the CSV data
DATA_FILE = "data/BTC-USDT_5m.csv"  # Update this to the path of your data file


def load_and_clean_data(filepath):
    """
    Load data from a CSV file, clean it, and prepare it for analysis.
    """
    # Load the data
    df = pd.read_csv(filepath)

    # Rename columns for readability
    df.rename(
        columns={
            "timestamp": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume_base": "Volume",
        },
        inplace=True,
    )

    # Drop unnecessary columns
    df.drop(columns=["trade_count", "turnover_quote", "is_complete"], inplace=True)

    # Convert timestamp to datetime and set as index
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)

    # Ensure all numeric columns are floats
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def visualize_with_matplotlib(df):
    """
    Visualize price and technical indicators using Matplotlib.
    """
    # Plot close price with SMA indicators
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["Close"], label="Close Price", color="blue", linewidth=1)

    # Add SMA indicators (if present)
    if "trend_sma_fast" in df.columns:
        plt.plot(df.index, df["trend_sma_fast"], label="SMA Fast", color="orange", linewidth=1)
    if "trend_sma_slow" in df.columns:
        plt.plot(df.index, df["trend_sma_slow"], label="SMA Slow", color="green", linewidth=1)

    # Add title, legend, and grid
    plt.title("BTC-USDT Price with SMA Indicators", fontsize=16)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("Price (USDT)", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    # Plot RSI (if present)
    if "momentum_rsi" in df.columns:
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df["momentum_rsi"], label="RSI", color="purple")
        plt.axhline(70, color="red", linestyle="--", label="Overbought")
        plt.axhline(30, color="green", linestyle="--", label="Oversold")
        plt.title("Relative Strength Index (RSI)", fontsize=16)
        plt.xlabel("Timestamp", fontsize=12)
        plt.ylabel("RSI", fontsize=12)
        plt.legend()
        plt.grid()
        plt.show()


def visualize_with_plotly(df):
    """
    Create an interactive candlestick chart with Plotly.
    """
    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="BTC-USDT"
    )])

    # Add SMA lines (if present)
    if "trend_sma_fast" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["trend_sma_fast"],
            mode="lines", line=dict(color="orange", width=1),
            name="SMA Fast"
        ))
    if "trend_sma_slow" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["trend_sma_slow"],
            mode="lines", line=dict(color="green", width=1),
            name="SMA Slow"
        ))

    # Update layout
    fig.update_layout(
        title="BTC-USDT Candlestick Chart with SMA",
        xaxis_title="Timestamp",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        showlegend=True
    )

    # Show the chart
    fig.show()


def main():
    """
    Main function to load, clean, and visualize the data.
    """
    # Load and clean the data
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_FILE)

    # Check if data was loaded successfully
    if df is not None:
        print("Data loaded successfully!")
        print(df.head())

        # Visualize with Matplotlib
        print("Visualizing with Matplotlib...")
        visualize_with_matplotlib(df)

        # Visualize with Plotly
        print("Visualizing with Plotly...")
        visualize_with_plotly(df)
    else:
        print("Failed to load data. Please check the file path and try again.")


if __name__ == "__main__":
    main()
