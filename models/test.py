import torch
import numpy as np
import gymnasium as gym
from dqn_agent import DQNAgent
import config
from envs.dogs_sheep_env import DogsSheepEnv

env = DogsSheepEnv(grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)
state_dim = env.observation_space.shape[0]
state, _ = env.reset()
action_dim = len(state)
agent = DQNAgent(state_dim, action_dim)
agent.model.load_state_dict(torch.load("output/dqn_model.pth"))
agent.model.eval()

for episode in range(10):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0

    for step in range(10):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        state = next_state
        total_reward += reward

        if done or truncated:
            break

    print(f"Test Episode {episode + 1}: Reward: {total_reward}")

env.close()
