import random
import torch
import torch.optim as optim
import torch.nn.functional as F

class QTrainer:
    def __init__(self, model, learning_rate, discount_factor, tau, batch_size):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.tau = tau
        self.batch_size = batch_size

    def tensorize_batch(self, batch):
        # create tensors
        states, actions, rewards, next_states, dones = batch
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32) # done flags 
        return states, actions, rewards, next_states, dones
    
        """ 
            tensors and datatypes taken from: 
            https://pytorch.org/docs/stable/tensors.html
            https://pytorch.org/docs/stable/generated/torch.cat.html
            https://www.geeksforgeeks.org/how-to-join-tensors-in-pytorch/

        """ 

    def calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = self.tensorize_batch(batch)
        # Compute Q-values for current and next states
        q_values = self.model(states)
        next_q_values = self.model(next_states)

        # Bellman equation from https://stackoverflow.com/questions/50581232/q-learning-equation-in-deep-q-network
        # Compute the target Q-values using the Bellman equation
        target_q_values = rewards + (1 - dones) * self.model.discount_factor * next_q_values.max(1)[0]
        predicted_q_values = q_values.gather(1, actions.unsqueeze(1))

        loss = F.smooth_l1_loss(predicted_q_values, target_q_values.unsqueeze(1))

        # MSE loss
        #loss = F.mse_loss(predicted_q_values, target_q_values)

        """ 
            loss taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html
            insight from https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
            insight from https://www.reddit.com/r/deeplearning/comments/iaw492/huber_loss_vs_mae_loss_in_dqn/
            insight from https://stats.stackexchange.com/questions/249355/how-exactly-to-compute-deep-q-learning-loss-function
            insight from https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

        """

        return loss

    def train_step(self, memory):
        # Sample random batch of experiences from the replay memory
        if len(memory) < self.batch_size:
            return

        batch = random.sample(memory, self.batch_size)
        loss = self.calculate_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, target_model):
        for target_param, param in zip(target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
