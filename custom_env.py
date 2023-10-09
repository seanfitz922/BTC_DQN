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
        super(CustomCryptoTradingEnv, self).__init__()

        num_time_steps = 1  # 1 hour (might change)
        num_features = len(df.columns)

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
        self.sma = 0.0  

        # Initialize state variables for tracking trades and profit
        self.num_trades = 0
        self.total_profit = 0.0

        # Initialize other state variables for steps
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.done = False


    def should_buy(self, close_price):
        # Determine the buy conditions (close price > SMA)
        return close_price > self.sma

    def should_sell(self, close_price):
        # Determine the sell conditions (close price < SMA)
        return close_price < self.sma

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
        """
        Execute the action, calculate reward, update state 
        return new observation, reward, done, and info
        """
        # Ensure the episode hasn't ended
        if self.done:
            raise Exception("Episode is already done. Call reset() to start a new episode.")

        # Get the close price for the current step
        close_price = self.df['close'][self.current_step]

        # Update the price history with the latest close price
        self.price_history.append(close_price)

        # Calculate SMA using the price history
        if len(self.price_history) == self.sma_window:
            self.sma = np.mean(self.price_history)

        # Default reward and done values
        reward = 0.0
        done = False

        # Buy action
        if action == Actions.BUY.value: 
            # Ensure the agent is not already holding
            if not self.holding_asset:  
                if self.should_buy(close_price):
                    self.execute_buy(close_price)
        # Sell action
        elif action == Actions.SELL.value:  
            # Ensure the agent is holding
            if self.holding_asset:  
                if self.should_sell(close_price):
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

        new_observation = self.df.iloc[self.current_step].values

        # check whether the episode is done
        done = self.done

        info = {
            # Number of trades made in the episode
            'trades': self.num_trades,  
            # Total profit obtained in the episode
            'total_profit': self.total_profit, 
        }

        return new_observation, reward, done, info

