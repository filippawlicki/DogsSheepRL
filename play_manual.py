import sys
import pygame
from envs.wolf_sheep_env import WolfSheepEnv
import config
from pyswip import Prolog


def check_connection(start, end):
  prolog = Prolog()
  prolog.consult("prolog/logic.pl")
  query = f"path({start}, {end}, X)"
  result = list(prolog.query(query))
  return len(result) > 0


start_point = 'a'
end_point = 'e'
if check_connection(start_point, end_point):
  print(f"There is a connection between {start_point} and {end_point}.")
else:
  print(f"There is no connection between {start_point} and {end_point}.")

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


env = WolfSheepEnv(grid_size=config.GRID_SIZE, num_wolves=config.NUM_WOLVES, num_sheep=config.NUM_SHEEP)
obs, _ = env.reset()

done = False
while not done:
  env.render()  # Call the render method to display the game state

  print("Use WASD to move. Press Q to quit.")

  # Poll for events using pygame
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      sys.exit()

  # Check for key presses
  action = handle_input()

  if action is not None:  # If a valid action is detected
    actions = [action] * env.num_wolves  # Apply the same action for all wolves
    obs, reward, done, _, _ = env.step(actions)

env.close()
