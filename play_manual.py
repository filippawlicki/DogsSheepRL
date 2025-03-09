import gym
import keyboard
from envs.wolf_sheep_env import WolfSheepEnv

env = WolfSheepEnv(grid_size=10, num_wolves=2, num_sheep=5)
obs, _ = env.reset()

key_to_action = {
  "w": 0,  # Move up
  "s": 1,  # Move down
  "a": 2,  # Move left
  "d": 3  # Move right
}

done = False
while not done:
  env.render()

  keys = keyboard.read_event(suppress=True).name  # Wait for key press
  action = [key_to_action.get(keys, 0)] * env.num_wolves  # Apply the same action to all wolves

  obs, reward, done, _, _ = env.step(action)

print("Game Over!")
env.close()
