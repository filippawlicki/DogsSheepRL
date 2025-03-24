import torch
import numpy as np
import gym
from dqn_agent import DQNAgent
import config

env = gym.make(config.ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
agent.model.load_state_dict(torch.load("good_models/size5_dog2_sheep1/dqn_model.pth"))
agent.model.eval()

for episode in range(10):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0

    for step in range(config.MAX_STEPS):
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
