from pathlib import Path
import torch
import numpy as np
from dqn_agent import DQNAgent
import config
from envs.dogs_sheep_env import DogsSheepEnv
import gym
from gym import envs
from gym.envs.registration import register
import matplotlib.pyplot as plt

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
episodes = 20000
max_steps = 1000
checkpoint_freq = 1000
print_freq = 50
batch_size = 64
lr = 0.0001
gamma = 0.95

# Environment initialization
env = gym.make('DogsSheep-v0', grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)

# Get an initial observation (list of all positions)
observation, _ = env.reset()

# Pass observation shape
state_dim = observation.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
print("Device:", device)

agent = DQNAgent(state_dim, config.NUM_DOGS, lr=lr, gamma=gamma, device=device)

rewards = []
losses = []

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0
    episode_reward = 0
    episode_loss = 0
    for step in range(max_steps):
        action = agent.select_action(state)
        #print("Action:", action)  # Debugging line

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        episode_reward += reward

        loss = agent.train(batch_size)
        if loss is not None:
            episode_loss += loss

        if done or truncated:
            break

    agent.update_target_model()
    rewards.append(episode_reward)
    losses.append(episode_loss)

    if (episode + 1) % print_freq == 0:
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Loss: {episode_loss}")

    # Checkpoint model
    if (episode + 1) % checkpoint_freq == 0:
        formatted_loss = f"{episode_loss:.2f}"
        torch.save(agent.model.state_dict(), f"{config.OUTPUT_DIR}/dqn_model_{episode + 1}_loss_{formatted_loss}_reward_{total_reward}.pth")

    # Gradually reduce epsilon (exploration)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay


# Save trained model
torch.save(agent.model.state_dict(), f"{config.OUTPUT_DIR}/dqn_model.pth")
env.close()

# Plot rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards")
plt.show()

# Plot losses
plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()