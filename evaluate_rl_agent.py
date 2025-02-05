import gymnasium as gym
import numpy as np
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from trading_env import TradingEnv  # Ensure it imports the latest env

# Define the model to evaluate
MODEL_PATH = "checkpoints/optuna_best_model.zip"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}\nEnsure training has completed and checkpoint exists.")
    exit()

# Load dataset for evaluation
df = pd.read_csv("data/evaluation_data.csv")
env = gym.make("TradingEnv-v0", df=df)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

# Load model
model = PPO.load(MODEL_PATH, env=env)

# Evaluate model performance
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"✅ Evaluation Complete:\nMean Reward: {mean_reward}, Std Reward: {std_reward}")

# Additional performance metrics
def calculate_sharpe_ratio(returns):
    if len(returns) < 2 or np.std(returns) == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

# Extract rewards from episode history
returns = env.get_episode_rewards()
sharpe_ratio = calculate_sharpe_ratio(returns)
print(f"📊 Sharpe Ratio: {sharpe_ratio}")
