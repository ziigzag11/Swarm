import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from stable_baselines3 import PPO
from trading_env import TradingEnv

# Load model & data
df = pd.read_csv("data/BTC-USD.csv")
env = TradingEnv(df)
model = PPO.load("ppo_trading_agent")

# Initialize session state
if "portfolio_values" not in st.session_state:
    st.session_state["portfolio_values"] = []
if "trade_history" not in st.session_state:
    st.session_state["trade_history"] = []
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

# Streamlit layout
st.title("📈 RL Trading Agent Live Dashboard")

portfolio_chart = st.empty()
trade_log = st.empty()
reward_chart = st.empty()

obs, _ = env.reset()
obs = np.array(obs, dtype=np.float32)

expected_shape = env.observation_space.shape[0]

# ✅ Ensure obs has the exact expected shape (11,)
obs = obs[:expected_shape]  # Trim excess if needed
obs = obs.squeeze()  # ✅ Remove batch dimension if present

done = False
while not done:
    print(f"Obs Before Predict: {obs.shape}, Expected: {env.observation_space.shape}")  # Debugging

    action, _ = model.predict(obs)  # ✅ Pass correct shape

    obs, reward, done, _, _ = env.step(action)
    obs = np.array(obs, dtype=np.float32)

    # ✅ Maintain correct shape for next step
    obs = obs[:expected_shape]  # Trim excess if needed
    obs = obs.squeeze()  # ✅ Remove batch dimension

# Update live portfolio value
    current_value = env.balance + (env.position * df.loc[env.current_step, "close"])
    st.session_state["portfolio_values"].append(current_value)

    # Track trade history
    if action == 1:
        st.session_state["trade_history"].append({"Step": st.session_state["current_step"], "Action": "BUY"})
    elif action == 0 and env.position > 0:
        st.session_state["trade_history"].append({"Step": st.session_state["current_step"], "Action": "SELL"})

    # Update dashboard
    with portfolio_chart.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state["portfolio_values"], mode="lines", name="Portfolio Value"))
        fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Time Step", yaxis_title="Value (USD)")
        st.plotly_chart(fig, use_container_width=True)

    with trade_log.container():
        trade_df = pd.DataFrame(st.session_state["trade_history"])
        st.write("📊 **Trade History**")
        st.dataframe(trade_df)

    with reward_chart.container():
        st.write("💰 **Recent Reward:**", round(reward, 2))

    # Add delay to simulate real-time updates
    time.sleep(0.5)

st.success("✅ Trading session complete!")



 

   