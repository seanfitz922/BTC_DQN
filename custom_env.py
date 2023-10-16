import gym
import gym.spaces as spaces
import numpy as np
from collections import deque
from enum import Enum

class Actions(Enum):
    BUY = 0  # Buy action
    SELL = 1  # Sell action
    HOLD = 2  # Hold action

class CustomCryptoTradingEnv(gym.Env):
    def __init__(self, df, sma_window=24):
        super(CustomCryptoTradingEnv, self).__init()

        num_time_steps = 1  # 1 hour (might change)
        num_features = 3  # SMA, Sharpe Ratio, BBP

        # Define observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_time_steps, num_features))

        # Define action space using the enum class
        self.action_space = spaces.Discrete(len(Actions))

        # Store the dataset
        self.df = df

        # Initialize state variables for tracking holdings
        self.holding_asset = False
        self.bought_price = 0.0

        # Initialize state variables for SMA calculation
        self.sma_window = sma_window
        self.price_history = deque(maxlen=self.sma_window)

        # Initialize state variables for tracking trades and profit
        self.num_trades = 0
        self.total_profit = 0.0

        # Initialize other state variables for steps
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.done = False

    def calculate_sma(self, data):
        return np.mean(data)

    def calculate_sharpe_ratio(self, returns):
        return np.mean(returns) / np.std(returns)

    def calculate_bbp(self, close_prices):
        sma = self.calculate_sma(close_prices)
        std = np.std(close_prices)
        ubb = sma + 2 * std
        lbb = sma - 2 * std
        bbp = (close_prices[-1] - lbb) / (ubb - lbb)
        return bbp

    def reset(self):
        # Reset the environment to its initial state and return the initial observation.
        
        self.current_step = 0
        self.done = False
        self.holding_asset = False
        self.bought_price = 0.0
        self.num_trades = 0
        self.total_profit = 0.0

        # Clear the price history
        self.price_history.clear()

        # Generate and return the initial observation
        initial_observation = self.df.iloc[self.current_step].values
        return initial_observation

    def execute_buy(self, close_price):
        # Execute the buy order
        self.holding_asset = True
        self.bought_price = close_price
        # Adjust portfolio and holdings
        self.num_trades += 1

    def execute_sell(self, close_price):
        # Execute the sell order
        self.holding_asset = False
        sold_price = close_price
        # Calculate and log the reward based on profit/loss
        reward = (sold_price - self.bought_price) / self.bought_price
        self.total_profit += reward
        self.num_trades += 1

    def step(self, action):
        # Ensure the episode hasn't ended
        if self.done:
            raise Exception("Episode is already done. Call reset() to start a new episode.")

        # Get the close price for the current step
        close_price = self.df['close'][self.current_step]

        # Update the price history with the latest close price
        self.price_history.append(close_price)

        # Default reward and done values
        reward = 0.0
        done = False

        # Calculate the additional indicators
        sma = self.calculate_sma(self.df['close'].values)
        returns = np.diff(self.df['close'].values)
        sr = self.calculate_sharpe_ratio(returns)
        bbp = self.calculate_bbp(self.df['close'].values)

        # Update the observation with additional indicators
        observation = np.array([sma, sr, bbp])

        # Buy action
        if action == Actions.BUY.value:
            # Ensure the agent is not already holding
            if not self.holding_asset:
                self.execute_buy(close_price)
        # Sell action
        elif action == Actions.SELL.value:
            # Ensure the agent is holding
            if self.holding_asset:
                self.execute_sell(close_price)
                # Profit as reward
                reward = (close_price - self.bought_price) / self.bought_price

        # Hold action (no reward)
        elif action == Actions.HOLD.value:
            reward = 0.0

        # Update the current step
        self.current_step += 1

        # Determine if the episode is done
        if self.current_step >= self.max_steps:
            self.done = True

        # Check whether the episode is done
        done = self.done

        info = {
            # Action taken in episode
            "action": action,
            # Number of trades made in the episode
            'trades': self.num_trades,
            # Total profit obtained in the episode
            'total_profit': self.total_profit,
            # Close price
            'close_price': close_price
        }

        return observation, reward, done, info
