import sys
import pygame
import torch
import numpy as np
from envs.dogs_sheep_env import DogsSheepEnv
import config
from dqn_model import DQN

# Create the environment
env = DogsSheepEnv(grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)

# Get the initial observation
obs, _ = env.reset()

state_dim = env.observation_space.shape[0]
model = DQN(state_dim, config.NUM_DOGS)
model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/dqn_model.pth"))
model.eval()


def select_action(model, state):
  """Selects an action based on the state using the trained model."""
  with torch.no_grad():  # No need to track gradients for inference
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = model(state_tensor)

    # Select the action with the highest Q-value for each dog
    actions = q_values.argmax(dim=2).squeeze().cpu().numpy()
    #print("Actions:", actions)

  return actions


print("Model is playing. Press Q to quit.")

done = False
while not done:
  env.render()

  # Select actions for each dog using the trained model
  actions = select_action(model, obs)
  actions = actions.astype(int)  # Convert to integers for the environment

  # Take a step in the environment with the selected actions
  obs, reward, done, _, _ = env.step(actions)

  # Poll for events using pygame
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      sys.exit()

env.close()
