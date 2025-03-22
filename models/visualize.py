import sys
import pygame
import torch
import numpy as np
from envs.dogs_sheep_env import DogsSheepEnv
import config
from train_alt import QNetwork as DQN

def decode_action(action_int, num_dogs):
    """
    Given the composite action (an integer), decode it into a list of
    individual actions for each dog. For N dogs and 4 moves per dog,
    the composite action is represented in base-4.
    """
    actions = []
    for _ in range(num_dogs):
        actions.append(action_int % 4)
        action_int //= 4
    return actions[::-1]

def process_observation(obs):
    """
    Convert the observation dictionary to a single flat float32 numpy array.
    The state consists of:
      - Dogs positions (num_dogs x 2)
      - Sheep positions (num_sheep x 2)
      - Target position (2)
    """
    return np.concatenate([
        obs["dogs"].flatten(),
        obs["sheep"].flatten(),
        obs["target"].flatten()
    ]).astype(np.float32)

# Create the environment
env = DogsSheepEnv(
    grid_size=config.GRID_SIZE,
    num_dogs=config.NUM_DOGS,
    num_sheep=config.NUM_SHEEP
)

# Get the initial observation and process it
obs, _ = env.reset()
state = process_observation(obs)

# Determine the state vector dimension:
# For dogs: num_dogs * 2, for sheep: num_sheep * 2, and target: 2.
state_dim = config.NUM_DOGS * 2 + config.NUM_SHEEP * 2 + 2
# Define the composite action space size (4^(num_dogs)).
action_dim = 4 ** config.NUM_DOGS

# Load the trained model
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/dqn_model_final.pth", map_location=torch.device("cpu")))
model.eval()

def select_action(model, state):
    """
    Selects an action based on the current state using the trained model.
    Returns a list of actions (one for each dog).
    """
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        composite_action = int(q_values.argmax(dim=-1).item())
        actions = decode_action(composite_action, config.NUM_DOGS)
        return actions

print("Model is playing. Press Q or close the window to quit.")

done = False
while not done:
    env.render()

    # Select actions using the trained model, then update state from the new observation.
    actions = select_action(model, state)
    obs, reward, done, truncated, info = env.step(actions)
    state = process_observation(obs)

    # Process pygame events so we can exit with Q or by closing the window.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

env.close()