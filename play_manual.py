import sys

import pygame

import config
from envs.dogs_sheep_env import DogsSheepEnv

key_to_action = {
  pygame.K_w: 0,  # Move up
  pygame.K_s: 1,  # Move down
  pygame.K_a: 2,  # Move left
  pygame.K_d: 3  # Move right
}


def handle_input():
  """Reads key press using pygame."""
  keys = pygame.key.get_pressed()  # Get the current state of all keys

  if keys[pygame.K_q]:  # Quit the game
    pygame.quit()
    sys.exit()

  action = None
  if keys[pygame.K_w]:
    action = 0  # Up
  elif keys[pygame.K_s]:
    action = 1  # Down
  elif keys[pygame.K_a]:
    action = 2  # Left
  elif keys[pygame.K_d]:
    action = 3  # Right
  return action


env = DogsSheepEnv(grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)
obs, _ = env.reset()

print("Use WASD to move. Press Q to quit.")
done = False
while not done:
  env.render()  # Call the render method to display the game state


  # Poll for events using pygame
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      sys.exit()

  # Check for key presses
  action = handle_input()

  if action is not None:  # If a valid action is detected
    actions = [action] * env.num_dogs  # Apply the same action for all dogs
    obs, reward, done, _, _ = env.step(actions)

env.close()
