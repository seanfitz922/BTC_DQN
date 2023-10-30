import numpy as np
import pandas as pd
from custom_env import CustomCryptoTradingEnv
from dqn_agent import DQNAgent
from dqn_model import DQN
from plot_rewards import plot_profit

"""
to-do:
    -(done) move hyperparameters to new file
    -(done) add stopping criteria instead of fixed 1000 episodes
    -(done) implement exploration rate decay for episilon
    -fix save model
    -actually use short memory
    -(done) implement BBP, SR
    -tune hyper parameters
    -ability to save trained model
    -add data logging as opposed to print(info)
    -organize files
"""

def train():
    # Load the minute-level data and create custom env
    columns_to_load = ['timestamp', 'close']
    df = pd.read_csv('bitcoin_2017_to_2023.csv', usecols=columns_to_load, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample the data to hourly frequency
    df = df.resample('1H').agg({'close': 'last'}).ffill()
    
    env = CustomCryptoTradingEnv(df.iloc[::-1])

    # Create and initialize DQN agent
    input_size = 4 # sma, sr, bbp, close price
    #input_size = len(hourly_df.column)  # open, high, low, close, volume
    output_size = env.action_space.n  # buy, sell, hold
    agent = DQNAgent(input_size, output_size)

    # Training loop
    num_episodes = 100_000
    # Desired profit threshold
    target_profit = 1_000_000 
    # Minimum acceptable balance
    min_balance = 1_000 
    # Initial balance
    initial_balance = 10_000
    target_update_frequency = 100

    # Create an empty list to store episode rewards
    episode_rewards = []

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0.0
        episode_balance = initial_balance  # Initialize episode balance

        # Update epsilon at the beginning of each episode
        agent.update_epsilon(episode)

        while not done:
            # Choose an action using your DQN agent's policy
            action = agent.select_action(observation)

            # Take a step in the environment
            new_observation, reward, done, info = env.step(action)

            # Store the experience in the agent's replay memory
            agent.remember(observation, action, reward, new_observation, done)

            # Train the agent using experience replay 
            agent.train_long_memory()

            # Update the current observation
            observation = new_observation

            episode_reward += reward
            episode_balance += reward

            # Stopping condition: stop if the agent achieves the target profit or goes below the minimum balance
            if episode_reward >= target_profit or episode_balance < min_balance:
                done = True
                print('Stop condition met')

            print(f"Action Taken: {info['action']}, Number of Trades: {info['trades']}, Close Price: {info['close_price']} Total Profit: {info['total_profit']:.2f}")

            # Append the episode reward to the list
            episode_rewards.append(episode_reward)

            # After a certain number of steps, update the target network
            if episode % target_update_frequency == 0:
                agent.update_target_network()

            # After training, call the plotting function from the separate file
            #plot_profit(episode_rewards) 

    agent.save('models/my_dqn_model.pth')

if __name__ == '__main__':
    train()
