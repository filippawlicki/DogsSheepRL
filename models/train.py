import torch
import numpy as np
from dqn_agent import DQNAgent
import config
from envs.dogs_sheep_env import DogsSheepEnv
import gym
from gym import envs
from gym.envs.registration import register
import matplotlib.pyplot as plt


env_id = 'DogsSheep-v0'

if env_id not in envs.registry:
    print(f"Registering {env_id}")
    register(
        id=env_id,
        entry_point='envs.dogs_sheep_env:DogsSheepEnv',
        max_episode_steps=1000,
    )

# Training hyperparameters
episodes = 100
max_steps = 1000

# Environment initialization
env = gym.make('DogsSheep-v0', grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)
state_dim = env.observation_space.shape[0]

print("State dimensions:", state_dim)

agent = DQNAgent(state_dim, config.NUM_DOGS)

rewards = []

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0
    episode_reward = 0
    for step in range(max_steps):
        # Select action for all dogs
        action = agent.select_action(state)
        #print("Action:", action)  # Debugging line

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        episode_reward += reward

        agent.train(config.BATCH_SIZE)

        if done or truncated:
            break

    agent.update_target_model()
    rewards.append(episode_reward)

    # Gradually reduce epsilon (exploration)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

# Save trained model
torch.save(agent.model.state_dict(), "dqn_model.pth")
env.close()

# Plot rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards")
plt.show()