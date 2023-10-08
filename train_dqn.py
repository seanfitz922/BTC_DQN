import numpy as np
import pandas as pd
from collections import deque
from custom_env import CustomCryptoTradingEnv
from dqn_agent import DQNAgent

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

    # resample and agg from https://stackoverflow.com/questions/47938372/pandas-dataframe-resample-aggregate-function-use-multiple-columns-with-a-customi
    # Resample the data to hourly intervals (OHLCV data)
    hourly_df = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    env = CustomCryptoTradingEnv(hourly_df)

    # Create and train DQN agent
    agent = DQNAgent(input_size='something', output_size='something')
    agent.train(batch_size=batch_size)


if __name__ == '__main__':
    train()