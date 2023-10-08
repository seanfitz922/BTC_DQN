import torch.optim as optim
from torch.distributions import Categorical

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr # learning rate, step size
        self.gamma = gamma # discount rate determining value of future rewards
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # optimization algorithm for training, updates weights
        self.criterion = nn.MSELoss() # measures loss between predicted q values and targeted q values (Mean Square Error)

