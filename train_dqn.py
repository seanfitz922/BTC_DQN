import numpy as np
import pandas as pd
from custom_env import CustomCryptoTradingEnv
from dqn_agent import DQNAgent

"""
to-do:
    -implement exploration rate decay for episilon
    -(done) add stopping criteria instead of fixed 1000 episodes
    -tune hyper parameters
    -ability to save trained model
    -add data logging as opposed to print(info)
"""

# Define hyperparameters
learning_rate = 0.001
discount_factor = 0.99
exploration_rate = 0.1
batch_size = 32
max_memory = 10000

def train():
    # Load the hourly data and create custom env
    columns_to_load = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv('bitcoin_2017_to_2023.csv', usecols=columns_to_load, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample the data to hourly intervals (OHLCV data)
    hourly_df = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    env = CustomCryptoTradingEnv(hourly_df)

    # Create and initialize DQN agent
    input_size = len(hourly_df.columns)  # open, high, low, close, volume
    output_size = len(env.action_space)  # buy, sell, hold
    agent = DQNAgent(input_size=input_size, output_size=output_size)

    # Training loop
    num_episodes = 1000  
    # Desired profit threshold
    target_profit = 1_000_000 
    # Minimum acceptable balance
    min_balance = 1_000 
    # Initial balance
    initial_balance = 10_000

    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0.0  # Initialize episode reward
        episode_balance = initial_balance  # Initialize episode balance

        while not done:
            # Choose an action using your DQN agent's policy
            action = agent.select_action(observation)

            # Take a step in the environment
            new_observation, reward, done, info = env.step(action)

            # Store the experience in the agent's replay memory
            agent.remember(observation, action, reward, new_observation, done)

            # Train the agent using experience replay 
            agent.train(batch_size=batch_size)

            # Update the current observation
            observation = new_observation

            episode_reward += reward
            episode_balance += reward 

            # Stopping condition: stop if the agent achieves the target profit or goes below the minimum balance
            if episode_reward >= target_profit or episode_balance < min_balance:
                done = True

            print(info)

if __name__ == '__main__':
    train()
