import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

class TradingEnv(gym.Env):
    """A trading environment for reinforcement learning with risk-based rewards and penalties."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, df, initial_balance=100, leverage=5, live_mode=False, render_mode="human"):
        super().__init__()
        
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.leverage = leverage  # Adjusted to Kraken's max leverage
        self.live_mode = live_mode  # Placeholder for live trading integration
        self.render_mode = render_mode  # ✅ Now supported in __init__
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for no position
        self.entry_price = 0
        self.portfolio_value = initial_balance
        self.r_factor = 2  # Reward-to-risk ratio enforcement
        self.trade_risk = 1  # Adjusted risk per trade to $1 (1% risk per trade)

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(11,), dtype=np.float32  # Ensure it matches training
        )


        # Validate dataset structure
        required_columns = {"open", "high", "low", "close", "volume"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"\u274c ERROR: Dataset is missing required columns! Available columns: {list(self.df.columns)}")

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.portfolio_value = self.initial_balance
        return self._next_observation(), {}

    def _next_observation(self):
        """Get the next observation from the data."""
        obs = self.df.iloc[self.current_step].drop("timestamp").values.astype(np.float32)
        return obs

    def step(self, action):
        """Take an action and return the new state, reward, and done flag."""
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self.df.iloc[self.current_step]["close"]
        reward = 0

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price

        elif action == 2:  # Sell
            if self.position == 1:
                profit = (current_price - self.entry_price) * 100  # Placeholder calculation
                reward = profit
                self.balance += profit
                self.position = 0

        self.portfolio_value = self.balance
        return self._next_observation(), reward, done, False, {}  # ✅ Fix unpacking issue

    def render(self):
        """Render the environment state."""
        if self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)  # ✅ Dummy RGB frame for rendering
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position} | Portfolio Value: {self.portfolio_value:.2f}")

    def close(self):
        """Close the environment."""
        pass
