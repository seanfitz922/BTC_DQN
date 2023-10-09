import random
import torch
from collections import deque
from dqn_model import DQN
from q_trainer import QTrainer
from config import exploration_rate, max_memory, learning_rate, discount_factor, tau, batch_size

class DQNAgent:
    def __init__(self, input_size, output_size, final_epsilon=0.1, epsilon_decay_steps=1000):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy weights/model parameters from policy to target
        self.target_net.eval()  # Set target network to evaluation mode (disables dropout and batch normalization)
        self.memory = deque(maxlen=max_memory)  # Replay memory buffer
        self.epsilon = exploration_rate
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        self.trainer = QTrainer(
            model=self.policy_net,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            tau=tau,
            batch_size=batch_size
        )

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
            state = torch.from_numpy(state).float()
            # no_grad() taken from https://pytorch.org/docs/stable/generated/torch.no_grad.html
            with torch.no_grad(): # disable gradient computation
                q_values = self.policy_net(state)
                # select action with highest Q-value
                return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in the replay memory
        
        self.memory.append((state, action, reward, next_state, done))
        #print("Experience added to memory:", (state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) < self.trainer.batch_size:
            return
        #print("Contents of memory before training:", self.memory)
        self.trainer.train_step(self.memory)


    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model using a single short memory (single step)
        self.trainer.train_step([(state, action, reward, next_state, done)])

    def update_target_network(self):
        # Update the target network by copying weights from the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())

