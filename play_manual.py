import sys
import os
import gym
from envs.wolf_sheep_env import WolfSheepEnv

key_to_action = {
    "w": 0,  # Move up
    "s": 1,  # Move down
    "a": 2,  # Move left
    "d": 3   # Move right
}

# Cross-platform key input
def get_key():
    """Reads a single key press."""
    if os.name == 'nt':  # For Windows
        import msvcrt
        return msvcrt.getch().decode("utf-8")
    else:  # For macOS/Linux
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

env = WolfSheepEnv(grid_size=10, num_wolves=2, num_sheep=5)
obs, _ = env.reset()

done = False
while not done:
    env.render()

    print("Use WASD to move. Press Q to quit.")
    key = get_key()

    if key.lower() == "q":
        break

    action = [key_to_action.get(key.lower(), 0)] * env.num_wolves
    obs, reward, done, _ = env.step(action)  # Unpack only 4 values

env.close()
