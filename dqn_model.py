import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Deep Q-Network model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define neural network architecture
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        # Forward pass 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    def save(self, file_name='models/my_dqn_model.pth'):
        # Save the model's state_dict to a file
        if file_name:
            torch.save(self.state_dict(), file_name)
