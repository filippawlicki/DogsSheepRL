import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_dim, num_dogs):
        super(DQN, self).__init__()
        self.num_dogs = num_dogs  # Store num_dogs as a class attribute
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Output shape: num_dogs * 4 (4 actions per dog)
        self.fc3 = nn.Linear(128, self.num_dogs * 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Output shape: (batch_size, num_dogs * 4)
        x = self.fc3(x)
        # Reshape to (batch_size, num_dogs, 4)
        return x.view(-1, self.num_dogs, 4)
