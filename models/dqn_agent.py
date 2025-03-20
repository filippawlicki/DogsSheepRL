import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import config
from models.dqn_model import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=config.LEARNING_RATE, gamma=config.GAMMA, device="cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = config.EPSILON
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=1000)

    def select_action(self, state):
        """Given a state, return an action for each dog."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)

        action = q_values.argmax(dim=-1).squeeze().cpu().numpy()  # Select max action for each dog
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        """Trains the neural network on samples from memory."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert batch to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(self.device)

        # Get Q-values for the current state and action
        model_output = self.model(states)

        q_values = model_output.gather(2, actions.unsqueeze(2))
        q_values = q_values.squeeze(2)

        next_q_values = self.target_model(next_states).max(2)[0].detach()

        # Calculate the target Q-values for the Bellman update
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_model(self):
        """Copy the model parameters to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())
