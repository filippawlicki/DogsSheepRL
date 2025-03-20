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
        """
        Given a state, return an action for each dog.
        Instead of selecting one action, we return an array of actions
        for each dog.
        """
        # Assuming the network outputs one action per dog
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor)  # This should output a tensor of shape (num_dogs, 4)

        # Convert q_values to actions
        # If q_values has shape (num_dogs, 4), we can directly use argmax along axis 1
        action = q_values.argmax(dim=-1).squeeze().cpu().numpy()  # Select max action for each dog
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Dodaje doświadczenie do pamięci."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        """Trains the neural network on samples from memory."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert batch to tensors
        states = torch.tensor(states, dtype=torch.float32)  # Shape: [batch_size, num_dogs, state_dim]
        actions = torch.tensor(actions, dtype=torch.int64)  # Shape: [batch_size, num_dogs]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)  # Shape: [batch_size, 1]
        next_states = torch.tensor(next_states, dtype=torch.float32)  # Shape: [batch_size, num_dogs, state_dim]
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)  # Shape: [batch_size, 1]

        # print("States shape:", states.shape)
        # print("Actions shape:", actions.shape)
        # print("Rewards shape:", rewards.shape)
        # print("Next states shape:", next_states.shape)
        # print("Dones shape:", dones.shape)

        # Get Q-values for the current state and action
        model_output = self.model(states)  # Shape: [batch_size, num_dogs, num_actions]
        #print("Model output shape:", model_output.shape)

        q_values = model_output.gather(2, actions.unsqueeze(2))  # Shape: [batch_size, num_dogs, 1]
        #print("Q-values shape before squeeze:", q_values.shape)
        #print("Q-values:", q_values)

        q_values = q_values.squeeze(2)  # Shape: [batch_size, num_dogs]
        #print("Q-values shape after squeeze:", q_values.shape)
        #print("Q-values after squeeze:", q_values)

        # Get the max Q-values for the next state (from target model)
        next_q_values = self.target_model(next_states).max(2)[0].detach()  # Shape: [batch_size, num_dogs]
        #print("Next Q-values shape:", next_q_values.shape)

        # Calculate the target Q-values for the Bellman update
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values  # Shape: [batch_size, num_dogs]
        #print("Target Q-values shape:", target_q_values.shape)

        # Compute the loss
        loss = self.loss_fn(q_values, target_q_values)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """Kopiuje wagi do modelu docelowego."""
        self.target_model.load_state_dict(self.model.state_dict())
