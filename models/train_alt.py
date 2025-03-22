import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gymnasium.envs.registration import register
from gymnasium import envs
import matplotlib.pyplot as plt
import os

# Import your configuration and environment.
import config

# -------------------------------------------------------------------
# Replay Buffer for storing transitions
# -------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------------
# Q-Network (a simple MLP)
# -------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------------------------------------------
# Helpers to process observations and decode composite actions
# -------------------------------------------------------------------
def process_observation(obs):
    """
    Convert the observation dictionary to a single flat float32 numpy array.
    State consists of:
      - Dogs positions (num_dogs x 2)
      - Sheep positions (num_sheep x 2)
      - Target position (2)
    """
    return np.concatenate([
        obs["dogs"].flatten(),
        obs["sheep"].flatten(),
        obs["target"].flatten()
    ]).astype(np.float32)

def decode_action(action_int, num_dogs):
    """
    Given a composite action (an integer), decode it into a list of actions (one per dog).
    With N dogs each having 4 directional moves, this function decomposes the integer
    into its base‑4 digits.
    """
    actions = []
    for _ in range(num_dogs):
        actions.append(action_int % 4)
        action_int //= 4
    return actions[::-1]

# -------------------------------------------------------------------
# Hyperparameters for training
# -------------------------------------------------------------------
EPISODES = 500             # Total episodes for training
MAX_STEPS = 500            # Maximum steps per episode
BATCH_SIZE = 64
GAMMA = 0.99               # Discount factor
LR = 1e-3                  # Learning rate
TARGET_UPDATE = 10         # Frequency (in episodes) to update target network
REPLAY_BUFFER_CAPACITY = 10000
EPS_START = 1.0            # Initial epsilon for epsilon-greedy strategy
EPS_END = 0.05             # Minimum epsilon
EPS_DECAY = 300            # Controls the decay rate of epsilon
CHECKPOINT_FREQ = 50       # Save model checkpoint every 50 episodes

# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def train():
    env_id = 'DogsSheep-v0'

    if env_id not in envs.registry:
        print(f"Registering {env_id}")
        register(
            id=env_id,
            entry_point='envs.dogs_sheep_env:DogsSheepEnv',
            max_episode_steps=1000,
        )
    env = gym.make('DogsSheep-v0', grid_size=config.GRID_SIZE,
                   num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)

    # Determine the state vector dimension.
    # For dogs: num_dogs * 2, for sheep: num_sheep * 2, and target: 2.
    state_dim = config.NUM_DOGS * 2 + config.NUM_SHEEP * 2 + 2

    # Define the composite action space size.
    # For each dog there are 4 moves; so total actions = 4^(num_dogs).
    action_dim = 4 ** config.NUM_DOGS

    # Set up devices and networks.
    device = torch.device("cpu")
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Set target network to evaluation mode

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    epsilon = EPS_START

    # Lists for logging rewards and losses
    episode_rewards = []
    episode_losses = []   # Average loss per episode
    all_losses = []       # All mini-batch losses across episodes

    for episode in range(EPISODES):
        # Reset the environment; note reset() returns (observation, info)
        obs, _ = env.reset()
        state = process_observation(obs)
        total_reward = 0
        losses_in_episode = []

        for t in range(MAX_STEPS):
            # --- Select action using epsilon-greedy ---
            if random.random() < epsilon:
                composite_action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    composite_action = int(q_values.argmax(dim=1).item())

            # Decode the composite action into a list (one action per dog)
            dog_actions = decode_action(composite_action, config.NUM_DOGS)

            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(dog_actions)
            next_state = process_observation(next_obs)
            total_reward += reward

            # Store the transition in replay buffer
            replay_buffer.push(state, composite_action, reward, next_state, done)
            state = next_state

            # --- Learning: sample a mini-batch and update QNetwork ---
            if len(replay_buffer) >= BATCH_SIZE:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)

                batch_state      = torch.from_numpy(batch_state).to(device)
                batch_next_state = torch.from_numpy(batch_next_state).to(device)
                batch_action     = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
                batch_reward     = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(device)
                batch_done       = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)

                # Current Q values for the performed actions
                current_q_values = policy_net(batch_state).gather(1, batch_action)

                # Compute the target Q value using the target network.
                with torch.no_grad():
                    max_next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    target_q_values = batch_reward + GAMMA * max_next_q_values * (1 - batch_done)

                loss = nn.MSELoss()(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log the loss for this mini-batch
                losses_in_episode.append(loss.item())
                all_losses.append(loss.item())

            if done:
                break

        # --- Epsilon decay ---
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)

        # Compute average loss for the episode (if any update occurred)
        avg_loss = np.mean(losses_in_episode) if losses_in_episode else 0.0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        # --- Periodically update the target network ---
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # --- Save checkpoints ---
        if (episode + 1) % CHECKPOINT_FREQ == 0:
            torch.save(policy_net.state_dict(), f"{config.OUTPUT_DIR}/dqn_model_episode_{episode+1}.pth")

        print(f"Episode {episode + 1:03d}/{EPISODES}, Total Reward: {total_reward:.2f}, "
              f"Average Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")

    # Save the final model at the end of training.
    torch.save(policy_net.state_dict(), f"{config.OUTPUT_DIR}/dqn_model_final.pth")
    print(f"Final model saved")
    env.close()
    print("Training complete.")

    # Plot the rewards and losses charts
    plt.figure(figsize=(12, 5))

    # Plot Episode Rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()

    # Plot Episode Losses
    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, label='Average Loss per Episode', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
if __name__ == "__main__":
    train()