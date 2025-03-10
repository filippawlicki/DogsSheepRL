import gym
import numpy as np
import random
import config
from .render import GameRenderer

class WolfSheepEnv(gym.Env):
    def __init__(self, grid_size=10, num_wolves=2, num_sheep=5):
        self.grid_size = grid_size
        self.num_wolves = num_wolves
        self.num_sheep = num_sheep

        # Agent positions
        self.wolves = [self._random_pos() for _ in range(self.num_wolves)]
        self.sheep = [self._random_pos() for _ in range(self.num_sheep)]
        self.target = self._random_pos()

        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_wolves)  # Every wolf can move in 4 directions

        obs_size = (self.num_wolves + self.num_sheep) * 2 + 2  # (x, y) for each wolf, sheep, and target
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(obs_size,), dtype=np.int32)

        self.renderer = GameRenderer(grid_size)

    def _random_pos(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Ensures Gym's reset behavior

        self.wolves = [self._random_pos() for _ in range(self.num_wolves)]
        self.sheep = [self._random_pos() for _ in range(self.num_sheep)]

        observation = self._get_observation()
        info = {}  # Gym requires this

        return observation, info

    def step(self, wolf_actions):
        """
        Makes a step in the environment.
        - Wolf actions is a list of actions for each wolf.
        - Returns observation, reward, done flag, and additional info.
        - Sheep move randomly, avoiding wolves and each other.
        """
        self._move_wolves(wolf_actions)
        self._move_sheep()

        done = self._check_done()
        reward = self._compute_reward()

        return self._get_observation(), reward, done, {}, {}

    def _move_wolves(self, actions):
        directions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        for i, action in enumerate(actions):
            move = directions[action]
            self.wolves[i][0] = np.clip(self.wolves[i][0] + move[0], 0, self.grid_size - 1)
            self.wolves[i][1] = np.clip(self.wolves[i][1] + move[1], 0, self.grid_size - 1)

    def _move_sheep(self):
        """Sheeps move randomly, avoiding wolves and each other."""
        for i in range(self.num_sheep):
            nearest_wolf = min(self.wolves, key=lambda w: self._distance(w, self.sheep[i]))

            # Calculate direction to the nearest wolf
            dx = self.sheep[i][0] - nearest_wolf[0]
            dy = self.sheep[i][1] - nearest_wolf[1]

            # Avoiding other sheeps (if they are too close)
            crowded_sheep = sum(1 for s in self.sheep if self._distance(s, self.sheep[i]) < config.MIN_DISTANCE_SHEEP)
            if crowded_sheep > 1:
                dx += random.choice([-1, 1])
                dy += random.choice([-1, 1])

            # Choosing the direction to move
            move_x = np.clip(dx, -1, 1)
            move_y = np.clip(dy, -1, 1)

            # Moving the sheep
            self.sheep[i][0] = np.clip(self.sheep[i][0] + move_x, 0, self.grid_size - 1)
            self.sheep[i][1] = np.clip(self.sheep[i][1] + move_y, 0, self.grid_size - 1)

    def _distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        """Returns a vector with positions of wolves, sheep, and target."""
        return np.array([*sum(self.wolves, []), *sum(self.sheep, []), *self.target])

    def _check_done(self):
        """Checks if all sheep reached the target or were captured."""
        all_sheep_captured = all(s in self.wolves for s in self.sheep)
        all_sheep_in_target = all(s == self.target for s in self.sheep)
        return all_sheep_captured or all_sheep_in_target

    def _compute_reward(self):
        """Counts points for herding sheep."""
        sheep_near_target = sum(1 for s in self.sheep if self._distance(s, self.target) < config.TARGET_RADIUS)
        sheep_captured = sum(1 for s in self.sheep if s in self.wolves)
        return sheep_near_target * config.SHEEP_NEAR_TARGET_REWARD - sheep_captured * config.SHEEP_CAPTURED_PENALTY

    def render(self):
        self.renderer.render_game(self.wolves, self.sheep, self.target)

    def close(self):
        self.renderer.close()
