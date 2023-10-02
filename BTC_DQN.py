import numpy as np
import pandas as pd
import gym
import gym.spaces as spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

# data set from https://www.kaggle.com/datasets/jkraak/bitcoin-price-dataset/

learning_rate = 0.001  
discount_factor = 0.99  
exploration_rate = 0.1  
batch_size = 32  # number of experiences sampled for each update
max_memory = 10000  


# Define custom env
class CustomCryptoTradingEnv(gym.Env):
    def __init__(self, df):
        super(CustomCryptoTradingEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(...))  
        self.action_space = spaces.Discrete(...) # buy, sell, hold? (short, long, etc) 
        self.df = df
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        # Reset env and return the init observation
        #return 

    def step(self, action):
        pass
        # Execute the action, calculate reward, update state, and return new observation, reward, done, and info
        # return new_observation, reward, self.done, {}


# Define Deep Q-Network model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define nn archic
        pass

    def forward(self, x):
        # Forward pass of your neural network
        pass

    def save(self, file_name = ''):
        pass

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr # learning rate, step size
        self.gamma = gamma # discount rate determining value of future rewards
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # optimization algorithm for training, updates weights
        self.criterion = nn.MSELoss() # measures loss between predicted q values and targeted q values (Mean Square Error)

# define agent (move to new file)
class DQNAgent:
    def __init__(self, input_size, output_size):
        self.game_count = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (must be smaller than one)
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.model = DQN('', 256, '')
        self.trainer = QTrainer(self.model, lr= learning_rate, gamma=self.gamma)

    def get_state(self):
        # get current state of stock/crpto?
        pass
    
    def select_action(self, state):
        # Implement epsil-greedy action selection
        pass
        # return decison

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY limit occurs
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

# Load data and create custom env
columns_to_load = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv('bitcoin_2017_to_2023.csv', usecols=columns_to_load)
df.rename(columns={'close': 'Close'}, inplace=True)

env = CustomCryptoTradingEnv(df)

# Create and train DQN agent
agent = DQNAgent(input_size='something', output_size='something') 
agent.train(batch_size = batch_size)

# plot and run simulations

def train(self):
    agent = DQNAgent()
    while True: 
        pass

if __name__ == '__main__':
    train()