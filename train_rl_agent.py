import gymnasium as gym
import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from trading_env import TradingEnv
import pandas as pd
# Load dataset
def make_env():
    """Creates and returns a trading environment with data loaded."""
    try:
        df = pd.read_csv("data/BTC-USD.csv")  # Ensure the path is correct
        print("✅ Data Loaded Successfully:", df.head())  # Debugging output
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        return None  # Handle failure gracefully

    return TradingEnv(df)

env = make_env()

if env is None:
    raise RuntimeError("Failed to create environment due to data loading error.")

# Use best hyperparameters found by Optuna
BEST_PARAMS = {
    "learning_rate": 0.0002771284466577116,
    "gamma": 0.9649923085051716,
    "gae_lambda": 0.9284830137722819,
    "batch_size": 512,
    "n_steps": 512,
    "ent_coef": 0.01646313577986376,
    "vf_coef": 0.7631628101446997,
}

# Create environment
env = make_env()

# Initialize model with best parameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=BEST_PARAMS["learning_rate"],
    gamma=BEST_PARAMS["gamma"],
    gae_lambda=BEST_PARAMS["gae_lambda"],
    batch_size=BEST_PARAMS["batch_size"],
    n_steps=BEST_PARAMS["n_steps"],
    ent_coef=BEST_PARAMS["ent_coef"],
    vf_coef=BEST_PARAMS["vf_coef"],
    verbose=1
)

# Train model
model.learn(total_timesteps=500_000)

# Save trained model
model.save("optimized_trading_agent")
print("✅ Training complete! Model saved as optimized_trading_agent.zip")
