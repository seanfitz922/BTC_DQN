import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from dqn_model import DQN
from config import learning_rate, discount_factor, exploration_rate, batch_size, max_memory

class DQNAgent:
    def __init__(self, input_size, output_size, final_epsilon=0.1, epsilon_decay_steps=1000):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy weights/model parameters from policy to target
        self.target_net.eval()  # Set target network to evaluation mode (disables dropout and batch normalization)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=max_memory)  # Replay memory buffer
        self.epsilon = exploration_rate
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        """
        target_net insight from: 
        https://stackoverflow.com/questions/54237327/why-is-a-target-network-required
        https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        https://pytorch.org/tutorials/beginner/saving_loading_models.html?highlight=eval

        optimizer taken from: 
        https://pytorch.org/docs/stable/optim.html 
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

        """

    def update_epsilon(self, episode):
        # Linear epsilon decay
        if episode < self.epsilon_decay_steps:
            # decay from https://stackoverflow.com/questions/53198503/epsilon-and-learning-rate-decay-in-epsilon-greedy-q-learning
            self.epsilon -= (self.epsilon - self.final_epsilon) / self.epsilon_decay_steps
        else:
            self.epsilon = self.final_epsilon

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(3)  # Explore: choose a random action (buy, sell, hold)
        else:
            # no_grad() taken from https://pytorch.org/docs/stable/generated/torch.no_grad.html
            with torch.no_grad(): # disable gradient computation
                q_values = self.policy_net(state)
                # select action with highest Q-value
                return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in the replay memory
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < batch_size:
            return

        # Sample random batch of experiences from the replay memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        """ 
        tensors and datatypes taken from: 
        https://pytorch.org/docs/stable/tensors.html
        https://pytorch.org/docs/stable/generated/torch.cat.html
        https://www.geeksforgeeks.org/how-to-join-tensors-in-pytorch/
        """ 
        # create tensors
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32) # done flags 

        # Compute Q-values for current and next states
        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states)

        # Bellman equation from https://stackoverflow.com/questions/50581232/q-learning-equation-in-deep-q-network
        # Compute the target Q-values using the Bellman equation
        target_q_values = rewards + (1 - dones) * discount_factor * next_q_values.max(1)[0]

        # Get the Q-values for the selected actions
        predicted_q_values = q_values.gather(1, actions.unsqueeze(1))

        """ 
            loss taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html
            insight from https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
            insight from https://www.reddit.com/r/deeplearning/comments/iaw492/huber_loss_vs_mae_loss_in_dqn/
            insight from https://stats.stackexchange.com/questions/249355/how-exactly-to-compute-deep-q-learning-loss-function
            insight from https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        """
        # Compute the loss
        loss = F.smooth_l1_loss(predicted_q_values, target_q_values.unsqueeze(1))

        # MSE loss
        #loss = F.mse_loss(predicted_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model using a single short memory (single step)
        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state)
        # Compute the target Q-values using the Bellman equation
        target_q_values = reward + (1 - done) * discount_factor * next_q_values.max(1)[0]
        predicted_q_values = q_values[0][action]

        # Compute the loss 
        loss = F.smooth_l1_loss(predicted_q_values, target_q_values)

        # MSE loss
        #loss = F.mse_loss(predicted_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        # Update the target network by copying weights from the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
