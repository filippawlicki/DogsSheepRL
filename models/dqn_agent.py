import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import config
from models.dqn_model import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=config.LEARNING_RATE, gamma=config.GAMMA):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = config.EPSILON
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())  # Kopiujemy wagi
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)

    def select_action(self, state):
        """Wybiera akcję zgodnie z polityką epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Eksploracja
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # Eksploatacja

    def store_transition(self, state, action, reward, next_state, done):
        """Dodaje doświadczenie do pamięci."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        """Trenuje sieć neuronową na próbkach z pamięci."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """Kopiuje wagi do modelu docelowego."""
        self.target_model.load_state_dict(self.model.state_dict())
