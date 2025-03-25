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
import time
import config

# Check if the output directory exists, if not, create it.
os.makedirs(f"{config.OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(f"{config.OUTPUT_DIR}/charts", exist_ok=True)

# Replay Buffer for storing transitions
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

# Q-Network (a simple MLP)
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Helpers to process observations and decode composite actions
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


# Hyperparameters for training
EPISODES = 1000000         # Total episodes for training
MAX_STEPS = 50         # Maximum steps per episode
BATCH_SIZE = 128
GAMMA = 0.99               # Discount factor
LR = 1e-4                  # Learning rate
TARGET_UPDATE = 25         # Frequency (in episodes) to update target network
REPLAY_BUFFER_CAPACITY = 50000
EPS_START = 1.5            # Initial epsilon for epsilon-greedy strategy
EPS_END = 0.01             # Minimum epsilon
EPS_DECAY = 1200            # Controls the decay rate of epsilon
CHECKPOINT_FREQ = 10000

# Main training loop
def save_plots(episode_rewards, episode_losses, episode):
    """ Save reward and loss plots as images. """
    plt.figure(figsize=(24, 8))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, label='Average Loss per Episode', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Episode')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/charts/training_progress_episode_{episode}.png")
    plt.close()



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
    action_dim = 4 * config.NUM_DOGS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up networks
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

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = process_observation(obs)
        total_reward = 0
        losses_in_episode = []

        for t in range(MAX_STEPS):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                actions = [random.randint(0, 3) for _ in range(config.NUM_DOGS)]
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor).cpu().numpy().flatten()
                    actions = [np.argmax(q_values[i*4:(i+1)*4]) for i in range(config.NUM_DOGS)]

            next_obs, reward, done, truncated, _ = env.step(actions)
            next_state = process_observation(next_obs)
            total_reward += reward

            # Store in replay buffer
            replay_buffer.push(state, actions, reward, next_state, done)
            state = next_state

            # Train only if buffer is ready
            if len(replay_buffer) >= BATCH_SIZE:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)

                batch_state = torch.from_numpy(batch_state).to(device)
                batch_next_state = torch.from_numpy(batch_next_state).to(device)
                batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)

                # Q-learning update
                current_q_values = policy_net(batch_state).gather(1, batch_action)
                with torch.no_grad():
                    max_next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    target_q_values = batch_reward + GAMMA * max_next_q_values * (1 - batch_done)

                loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()

                losses_in_episode.append(loss.item())

            if done or truncated:
                break


    # Epsilon decay
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)

        # Compute average loss for the episode (if any updates occurred)
        avg_loss = np.mean(losses_in_episode) if losses_in_episode else 0.0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        # Periodically update the target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save checkpoints
        if (episode + 1) % CHECKPOINT_FREQ == 0:
            torch.save(policy_net.state_dict(), f"{config.OUTPUT_DIR}/models/dqn_model_episode_{episode+1}.pth")
            save_plots(episode_rewards, episode_losses, episode+1)
            print(f"Episode {episode + 1:03d}/{EPISODES}, Total Reward: {total_reward:.2f}, "
            f"Average Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")

    env.close()
    # Save the final model
    torch.save(policy_net.state_dict(), f"{config.OUTPUT_DIR}/models/dqn_model_{config.GRID_SIZE}x{config.GRID_SIZE}+{config.NUM_DOGS}d+{config.NUM_SHEEP}o.pth")
    print("Training complete.")

if __name__ == "__main__":
    start = time.time()
    train()
    end = time.time()
    print(f"Training took {end - start:.2f} seconds.")