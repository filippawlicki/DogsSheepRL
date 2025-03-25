import sys
import pygame
import torch
import numpy as np
from envs.dogs_sheep_env import DogsSheepEnv
import config
from train_alt import QNetwork as DQN


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


env = DogsSheepEnv(grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)

# Get an initial observation (list of all positions)
obs, _ = env.reset()
state = process_observation(obs)

state_dim = config.NUM_DOGS * 2 + config.NUM_SHEEP * 2 + 2
action_dim = 4 ** config.NUM_DOGS  # Złożona przestrzeń akcji dla N psów

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


print("Model is playing. Press Q or close the window to stop the visualization.")
while True:
    env.reset()
    done = False
    while not done:
        env.render()

        actions = select_action(model, state)

        # Make a step in the environment
        obs, reward, done, truncated, _ = env.step(actions)
        state = process_observation(obs)

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

env.close()