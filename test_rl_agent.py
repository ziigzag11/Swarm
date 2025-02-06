import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from trading_env import TradingEnv
import os
import pandas as pd

# ✅ Define model paths
MODEL_PATHS = {
    "Optuna Best Model": "checkpoints/optuna_best_model.zip",
    "Latest Checkpoint": "checkpoints/latest_checkpoint.zip",
    "Optimized Trading Agent": "optimized_trading_agent.zip",
}

# ✅ Load dataset
DATA_FILE = "data/BTC-USD.csv"
df = pd.read_csv(DATA_FILE)

# ✅ Initialize trading environment with correct rendering
env = TradingEnv(df, render_mode="rgb_array")  # ✅ FIXED

# ✅ Gymnasium-compatible logging
env = gym.wrappers.RecordEpisodeStatistics(env)  # Tracks rewards & episode stats
env = gym.wrappers.RecordVideo(env, video_folder="./logs/videos", episode_trigger=lambda x: x % 5 == 0)

def test_model(model_path, model_name):
    """Load a trained model and run a test backtest."""
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_name}: File not found ({model_path})")
        return
    
    print(f"🚀 Testing {model_name} ({model_path})")
    
    # Load trained model
    model = PPO.load(model_path, env=env)
    
    obs, _ = env.reset()  # ✅ FIXED: Gymnasium now returns (obs, info)
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()  # ✅ Now supports video rendering

    print(f"🔎 {model_name} Backtest Completed. Total Reward: {total_reward:.2f}")

# ✅ Test all models sequentially
for model_name, model_path in MODEL_PATHS.items():
    test_model(model_path, model_name)
