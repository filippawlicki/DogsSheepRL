from pathlib import Path
import torch
import numpy as np
from dqn_agent import DQNAgent
import config
import gymnasium as gym
from gymnasium import envs
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import matplotlib.pyplot as plt
from scipy.stats import zscore
import logging

Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

env_id = 'DogsSheep-v0'

if env_id not in envs.registry:
    print(f"Registering {env_id}")
    register(
        id=env_id,
        entry_point='envs.dogs_sheep_env:DogsSheepEnv',
        max_episode_steps=1000,
    )

# Training hyperparameters
episodes = 10000
max_steps = config.MAX_EPISODE_STEPS
checkpoint_freq = 50
print_freq = 50
batch_size = 64
lr = 0.0005
gamma = 0.99
epsilon_greedy = True
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 1 - 1e-5

# Environment initialization
env = gym.make('DogsSheep-v0', grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)

# Get an initial observation (list of all positions)
observation, _ = env.reset()

# Pass observation shape
state_dim = observation.shape[0]

device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

agent = DQNAgent(state_dim, config.NUM_DOGS, lr=lr, gamma=gamma, device=device, epsilon_greedy=epsilon_greedy, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)

rewards = []
losses = []
epsilons = []

# Variables to track best model based on reward or loss
best_reward = -float('inf')
best_loss = float('inf')
best_reward_model = None
best_loss_model = None

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0
    episode_reward = 0
    episode_loss = 0
    for step in range(max_steps):
        #env.render()
        action = agent.select_action(state)
        #print("Action:", action)  # Debugging line

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        episode_reward += reward

        if done or truncated:
            break

    loss = agent.train(batch_size)
    if loss is not None:
        episode_loss += loss
    agent.update_target_model()
    rewards.append(episode_reward)
    losses.append(episode_loss)
    epsilons.append(agent.epsilon)

    if (episode + 1) % print_freq == 0:
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Loss: {episode_loss}, Epsilon: {agent.epsilon}")

    # Save best model based on reward
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_reward_model = agent.model.state_dict()  # Save the model's state_dict
        formatted_loss = f"{episode_loss:.2f}"
        torch.save(agent.model.state_dict(), f"{config.OUTPUT_DIR}/best_reward_model.pth")
        print(f"Best reward model saved at episode {episode + 1} - Reward: {best_reward}, Loss: {episode_loss}")

    # Checkpoint model
    if (episode + 1) % checkpoint_freq == 0:
        formatted_loss = f"{episode_loss:.2f}"
        checkpoint_data = {
            'episode': episode + 1,
            'model_state_dict': agent.model.state_dict(),
            'reward': total_reward,
            'loss': episode_loss,
            'epsilon': agent.epsilon
        }
        torch.save(checkpoint_data,
                   f"{config.OUTPUT_DIR}/checkpoint_{episode + 1}_loss_{formatted_loss}_reward_{total_reward}.pth")

    # Gradually reduce epsilon (exploration)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay


# Save trained model
torch.save(agent.model.state_dict(), f"{config.OUTPUT_DIR}/dqn_model.pth")
env.close()

# Remove outliers based on z-score threshold
z_score_threshold = 3  # Adjust threshold as needed
reward_z_scores = zscore(rewards)
loss_z_scores = zscore(losses)

# Filter out episodes with extreme rewards or losses
filtered_rewards = [reward for reward, z in zip(rewards, reward_z_scores) if abs(z) < z_score_threshold]
filtered_losses = [loss for loss, z in zip(losses, loss_z_scores) if abs(z) < z_score_threshold]
filtered_epsilons = [epsilon for epsilon, z in zip(epsilons, reward_z_scores) if abs(z) < z_score_threshold]

# Plot rewards (after outlier removal)
plt.plot(filtered_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards (Filtered)")
plt.savefig(f"{config.OUTPUT_DIR}/training_rewards_filtered.png")
plt.show()

# Plot losses with smoothing and outlier removal
window_size = 10  # Smoothing window size
smoothed_losses = np.convolve(filtered_losses, np.ones(window_size)/window_size, mode='valid')
plt.plot(smoothed_losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss (Smoothed, Filtered)")
plt.savefig(f"{config.OUTPUT_DIR}/training_losses_filtered.png")
plt.show()

# Plot epsilon decay (filtered)
plt.plot(filtered_epsilons)
plt.xlabel("Episode")
plt.ylabel("Epsilon (Exploration Rate)")
plt.title("Epsilon Decay Over Training (Filtered)")
plt.savefig(f"{config.OUTPUT_DIR}/epsilon_decay_filtered.png")
plt.show()