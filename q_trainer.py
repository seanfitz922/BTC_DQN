import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class QTrainer:
    def __init__(self, model, learning_rate, discount_factor, tau, batch_size):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.tau = tau
        self.batch_size = batch_size

    # def tensorize_batch(self, batch):
    #     # create tensors
    #     states, actions, rewards, next_states, dones = batch
    #     states = torch.cat(states)
    #     next_states = torch.cat(next_states)
    #     actions = torch.tensor(actions, dtype=torch.int64)
    #     rewards = torch.tensor(rewards, dtype=torch.float32)
    #     dones = torch.tensor(dones, dtype=torch.float32) # done flags 
    #     return states, actions, rewards, next_states, dones
    
        """ 
            tensors and datatypes taken from: 
            https://pytorch.org/docs/stable/tensors.html
            https://pytorch.org/docs/stable/generated/torch.cat.html
            https://www.geeksforgeeks.org/how-to-join-tensors-in-pytorch/

        """ 

    def calculate_loss(self, batch):
        
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert the elements of the batch to PyTorch tensors directly
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        

        # Bellman equation from https://stackoverflow.com/questions/50581232/q-learning-equation-in-deep-q-network
        # Compute the target Q-values using the Bellman equation
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values.max(1)[0]
        predicted_q_values = q_values.gather(1, actions.unsqueeze(1))

        # Calculate the smooth L1 loss
        loss = F.smooth_l1_loss(predicted_q_values, target_q_values.unsqueeze(1))
        
        return loss

        # MSE loss
        #loss = F.mse_loss(predicted_q_values, target_q_values)

        """ 
            loss taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html
            insight from https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
            insight from https://www.reddit.com/r/deeplearning/comments/iaw492/huber_loss_vs_mae_loss_in_dqn/
            insight from https://stats.stackexchange.com/questions/249355/how-exactly-to-compute-deep-q-learning-loss-function
            insight from https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

        """

    def train_step(self, memory):
        # Sample random batch of experiences from the replay memory
        if len(memory) < self.batch_size:
            return

        batch = random.sample(memory, self.batch_size)
        loss = self.calculate_loss(batch)  # Pass the entire batch to calculate_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

