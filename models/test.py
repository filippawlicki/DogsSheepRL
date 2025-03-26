import sys
import pygame
import torch
import numpy as np
from envs.dogs_sheep_env import DogsSheepEnv
import config
from train import QNetwork as DQN
import matplotlib.pyplot as plt
import pandas as pd


def decode_action(action_int, num_dogs):
    """
      Decodes a composite action (as a single integer) into a list of actions (one per dog).
      With N dogs each having 4 directional moves (0: up, 1: down, 2: left, 3: right),
      this function decomposes the integer into its base-4 digits.
      """
    actions = []
    for _ in range(num_dogs):
        actions.append(action_int % 4)
        action_int //= 4
    return actions[::-1]


def process_observation(obs):
    """
      Transform the observation dictionary to a single flat float32 numpy array.
      State consists of:
        - Dogs positions (num_dogs x 2),
        - Sheep positions (num_sheep x 2),
        - Target position (2).
      """
    return np.concatenate([
        obs["dogs"].flatten(),
        obs["sheep"].flatten(),
        obs["target"].flatten()
    ]).astype(np.float32)


state_dim = config.NUM_DOGS * 2 + config.NUM_SHEEP * 2 + 2
action_dim = 4 ** config.NUM_DOGS  # Composite action space for N dogs

# Load the trained model
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/dqn_model_5x5+2d+2o.pth", map_location="cpu"))
model.eval()


def select_action(model, state):
    """
      Based on the current state, selects an action using the trained model.
      Returns a list of actions, i.e., one move for each dog.
      """
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        composite_action = int(q_values.argmax(dim=-1).item())
        return decode_action(composite_action, config.NUM_DOGS)


MAX_STEPS = 100
MAP_SIZES = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
]
GAMES_PER_SIZE = 100

grids_statistics = {
    grid_size: {
        "average_reward_per_step": 0,
        "average_steps": 0,
        "won_games": 0,
        "games": []
    }
    for grid_size in MAP_SIZES
}

for grid_size in MAP_SIZES:
    total_rewards = 0
    total_steps = 0
    for _ in range(GAMES_PER_SIZE):
        rewards = 0
        steps_this_game = 0
        env = DogsSheepEnv(grid_size=grid_size, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)

        # Get an initial observation (list of all positions)
        obs, _ = env.reset()
        state = process_observation(obs)

        done = False
        while not done and steps_this_game < MAX_STEPS:
            steps_this_game += 1
            actions = select_action(model, state)

            # Make a step in the environment
            obs, reward, done, truncated, _ = env.step(actions)
            rewards += reward
            state = process_observation(obs)

        total_rewards += rewards
        total_steps += steps_this_game
        grids_statistics[grid_size]["games"].append({
            "average_reward_per_step": rewards / steps_this_game if steps_this_game > 0 else 0
        })

        if done:
            grids_statistics[grid_size]["won_games"] += 1

    grids_statistics[grid_size]["average_reward_per_step"] = total_rewards / total_steps if total_steps > 0 else 0
    grids_statistics[grid_size]["average_steps"] = total_steps / GAMES_PER_SIZE
    print(f"Grid {grid_size}x{grid_size}: win rate: {grids_statistics[grid_size]['won_games'] / GAMES_PER_SIZE}, ")

env.close()

# Plot the win rate for each grid size
plt.figure(figsize=(12, 6))
plt.plot(MAP_SIZES, [grids_statistics[grid_size]["won_games"] / GAMES_PER_SIZE for grid_size in MAP_SIZES], marker='o')
plt.xlabel('Grid Size')
plt.ylabel('Win Rate')
plt.title('Win Rate for Different Grid Sizes')
plt.grid()
plt.savefig(f"{config.OUTPUT_DIR}/win_rate_per_grid_size.png")
plt.show()

# Plot the average reward per step for each grid size and average steps per game on one plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(MAP_SIZES, [grids_statistics[grid_size]["average_reward_per_step"] for grid_size in MAP_SIZES], marker='o')
plt.xlabel('Grid Size')
plt.ylabel('Average Reward per Step')
plt.title('Average Reward per Step for Different Grid Sizes')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(MAP_SIZES, [grids_statistics[grid_size]["average_steps"] for grid_size in MAP_SIZES], marker='o')
plt.xlabel('Grid Size')
plt.ylabel('Average Steps per Game')
plt.title('Average Steps per Game for Different Grid Sizes')
plt.grid()
plt.savefig(f"{config.OUTPUT_DIR}/average_steps_and_reward_per_grid_size.png")
plt.show()

# Transform the grids_statistics dictionary to a DataFrame
data = []
for grid_size, stats in grids_statistics.items():
    for game in stats["games"]:
        data.append({
            "grid_size": grid_size,
            "average_reward_per_step": game["average_reward_per_step"],
            "average_steps": stats["average_steps"],
            "won_games": stats["won_games"]
        })

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(f"{config.OUTPUT_DIR}/grid_statistics.csv", index=False)

