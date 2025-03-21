import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_channels, grid_size, num_dogs, hidden_dim=256):
        super(DQN, self).__init__()
        self.num_dogs = num_dogs

        # Convolutional layers to process spatial data
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Flatten size calculation
        flattened_size = 128 * grid_size * grid_size  # Assuming full grid processing

        # Fully connected layers for action selection
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_dogs * 4)  # 4 actions per dog

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, self.num_dogs, 4)  # (batch, num_dogs, 4 actions)
