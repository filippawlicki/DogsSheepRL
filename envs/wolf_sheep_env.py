import gym
import numpy as np
import random

class WolfSheepEnv(gym.Env):
    def __init__(self, grid_size=10, num_wolves=2, num_sheep=5):
        self.grid_size = grid_size
        self.num_wolves = num_wolves
        self.num_sheep = num_sheep

        # Agent positions
        self.wolves = [self._random_pos() for _ in range(num_wolves)]
        self.sheep = [self._random_pos() for _ in range(num_sheep)]
        self.target = self._random_pos()

        self.action_space = gym.spaces.MultiDiscrete([4] * num_wolves)  # Every wolf can move in 4 directions

        obs_size = (num_wolves + num_sheep) * 2 + 2  # (x, y) for each wolf, sheep and target
        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(obs_size,), dtype=np.int32)

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
        - Returns observation, reward, done flag and additional info.
        - Sheep move randomly, avoiding wolves and each other.
        """
        self._move_wolves(wolf_actions)
        self._move_sheep()

        done = self._check_done()
        reward = self._compute_reward()

        return self._get_observation(), reward, done, {}

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
        """Sheep move towards the target, avoiding wolves and each other."""
        for i in range(self.num_sheep):
            target_dx = self.target[0] - self.sheep[i][0]
            target_dy = self.target[1] - self.sheep[i][1]

            # Avoiding wolves
            nearest_wolf = min(self.wolves, key=lambda w: self._distance(w, self.sheep[i]))
            wolf_dx = self.sheep[i][0] - nearest_wolf[0]
            wolf_dy = self.sheep[i][1] - nearest_wolf[1]

            # Avoiding other sheep (if they are too close)
            crowded_sheep = sum(1 for s in self.sheep if self._distance(s, self.sheep[i]) < 2)
            if crowded_sheep > 1:
                target_dx += random.choice([-1, 1])
                target_dy += random.choice([-1, 1])

            # Combining the directions: moving towards the target and away from wolves
            dx = np.clip(target_dx + np.sign(wolf_dx), -1, 1)
            dy = np.clip(target_dy + np.sign(wolf_dy), -1, 1)

            # Moving the sheep
            self.sheep[i][0] = np.clip(self.sheep[i][0] + dx, 0, self.grid_size - 1)
            self.sheep[i][1] = np.clip(self.sheep[i][1] + dy, 0, self.grid_size - 1)

    def _distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        """Returns a vector with positions of wolves, sheep and target."""
        return np.array([*sum(self.wolves, []), *sum(self.sheep, []), *self.target])

    def _check_done(self):
        """Checks if all sheep reached the target or were captured."""
        all_sheep_captured = all(s in self.wolves for s in self.sheep)
        all_sheep_in_target = all(s == self.target for s in self.sheep)
        return all_sheep_captured or all_sheep_in_target

    def _compute_reward(self):
        """Counts points for herding sheep."""
        sheep_near_target = sum(1 for s in self.sheep if self._distance(s, self.target) < 2)
        sheep_captured = sum(1 for s in self.sheep if s in self.wolves)
        return sheep_near_target * 5 - sheep_captured * 3

    def render(self):
        """Prints the current state of the simulation."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        for w in self.wolves:
            grid[w[0], w[1]] = "W"
        for s in self.sheep:
            grid[s[0], s[1]] = "S"
        grid[self.target[0], self.target[1]] = "T"

        print("\n".join(" ".join(row) for row in grid))
        print()
