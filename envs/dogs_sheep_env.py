import gymnasium as gym
import numpy as np
import random

from math import trunc

import config
from envs.render import GameRenderer
from pyswip import Prolog
import math
from pathlib import Path

prolog_file = str(config.ROOT_DIR / "prolog" / "logic.pl").replace("\\", "/")

prolog = Prolog()
prolog.consult(prolog_file)


class DogsSheepEnv(gym.Env):
    def __init__(self, grid_size=10, num_dogs=2, num_sheep=5):
        self.grid_size = grid_size
        self.num_dogs = num_dogs
        self.num_sheep = num_sheep

        # Agent positions
        self.dogs = [self._random_pos() for _ in range(self.num_dogs)]
        self.sheep = [self._random_pos() for _ in range(self.num_sheep)]
        self.target = self._random_pos()  # Target position

        self.prev_dogs = [dog[:] for dog in self.dogs]  # Previous dog positions
        self.prev_sheep = [sheep[:] for sheep in self.sheep]  # Previous sheep positions

        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_dogs)  # Every dog can move in 4 directions

        obs_size = (self.num_dogs + self.num_sheep) * 2 + 2  # (x, y) for each dog, sheep, and target
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(obs_size,), dtype=np.int32)

        self.renderer = GameRenderer(grid_size)

        # Send target information to Prolog
        self._send_target_to_prolog()

    def _send_target_to_prolog(self):
        prolog.retractall("target_position(_, _)")  # Clear existing target position
        prolog.assertz(f"target_position({self.target[0]}, {self.target[1]})")

    def _random_pos(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dogs = [self._random_pos() for _ in range(self.num_dogs)]
        self.sheep = [self._random_pos() for _ in range(self.num_sheep)]
        self.target = self._random_pos()

        self.prev_dogs = [dog[:] for dog in self.dogs]
        self.prev_sheep = [sheep[:] for sheep in self.sheep]

        self._send_target_to_prolog()
        observation = self._get_observation()
        return observation, {}

    def step(self, dog_actions):
        """
        Makes a step in the environment.
        - Dog actions is a list of actions for each dog.
        - Returns observation, reward, done flag, truncated flag.
        - Sheep avoide dogs.
        """
        self._move_dogs(dog_actions)
        self._move_sheep()

        done = self._check_done()
        truncated = False  # Not used in this environment
        reward = self._compute_reward()

        return self._get_observation(), reward, bool(done), bool(truncated), {}

    def _move_dogs(self, actions):
        directions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        for i, action in enumerate(actions):
            move = directions[action]
            self.dogs[i][0] = np.clip(self.dogs[i][0] + move[0], 0, self.grid_size - 1)
            self.dogs[i][1] = np.clip(self.dogs[i][1] + move[1], 0, self.grid_size - 1)

    def _move_sheep(self):
        """Sheeps move randomly, avoiding dogs and each other."""
        for i in range(self.num_sheep):
            # Calculate crowdedness
            crowded_sheep = sum(1 for s in self.sheep if self._distance(s, self.sheep[i]) < config.MIN_DISTANCE_SHEEP)

            # Convert dog positions to Prolog format
            dogs_prolog = "[" + ", ".join(f"({dog[0]}, {dog[1]})" for dog in self.dogs) + "]"
            sheep_prolog = "[" + ", ".join(f"({sheep[0]}, {sheep[1]})" for sheep in self.sheep) + "]"

            # Calculate direction to the nearest dog using Prolog
            query = f"direction_to_move({self.sheep[i][0]}, {self.sheep[i][1]}, {dogs_prolog}, {crowded_sheep}, {config.SHEEP_VISION_RANGE}, {sheep_prolog}, MoveX, MoveY)"
            #print(f"\nProlog query: {query}")  # Debugging line
            result = list(prolog.query(query))
            #print("target: ", list(prolog.query(f"target_position(X, Y)"))[0])
            #print("result: ", result)
            if not result:
                #print(f"No result for query: {query}")  # Debugging line
                move_x, move_y = 0, 0  # Default move if no result
            else:
                try:
                    move_x = result[0]['MoveX']
                    move_y = result[0]['MoveY']
                except (ValueError, KeyError) as e:
                    print(f"Error converting Prolog result to int: {e}")  # Debugging line
                    move_x, move_y = 0, 0  # Default move if conversion fails
            #print(f"move_x: {move_x}, move_y: {move_y}")  # Debugging line


            # Moving the sheep
            self.sheep[i][0] = np.clip(self.sheep[i][0] + move_x, 0, self.grid_size - 1)
            self.sheep[i][1] = np.clip(self.sheep[i][1] + move_y, 0, self.grid_size - 1)

    def _distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _get_observation(self):
        """Returns a multi-channel observation tensor representing the environment."""

        # Initialize multi-channel observation
        obs = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 1: Dog positions (binary mask)
        for x, y in self.dogs:
            obs[0, x, y] = 1

            # Channel 2: Sheep positions (binary mask)
        for x, y in self.sheep:
            obs[1, x, y] = 1

            # Channel 3: Target position (binary mask)
        tx, ty = self.target
        obs[2, tx, ty] = 1

        # Channel 4: Distance to nearest sheep for each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                min_dist = min(np.linalg.norm([i - sx, j - sy]) for sx, sy in self.sheep)
                obs[3, i, j] = min_dist / self.grid_size  # Normalize to [0, 1]

        # Channel 5: Distance to target for each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.linalg.norm([i - tx, j - ty])
                obs[4, i, j] = dist / self.grid_size  # Normalize to [0, 1]

        return obs  # Shape: (5, grid_size, grid_size)

    def _check_done(self):
        """Checks if all sheep reached the target or were captured."""
        all_sheep_captured = all(s in self.dogs for s in self.sheep)
        all_sheep_in_target = all(s == self.target for s in self.sheep)
        return all_sheep_captured or all_sheep_in_target

    def _compute_reward(self):
        """Compute the reward for the current state."""
        total_reward = 0

        # Reward for moving dogs, to encourage them to move
        for i, dog in enumerate(self.dogs):
            if dog != self.prev_dogs[i]:
                total_reward += config.DOG_MOVE_REWARD


        # Reward for sheep moving toward target
        for i, sheep in enumerate(self.sheep):
            old_distance = self._distance(self.prev_sheep[i], self.target)
            new_distance = self._distance(sheep, self.target)

            if new_distance < old_distance:
                total_reward += config.SHEEP_MOVE_TOWARD_TARGET_REWARD

            if new_distance < 1 and new_distance < old_distance:
                total_reward += config.SHEEP_CAPTURED_REWARD

        # Reward for reducing the max sheep distance from the target
        old_max_dist = max(self._distance(s, self.target) for s in self.prev_sheep)
        new_max_dist = max(self._distance(s, self.target) for s in self.sheep)

        if new_max_dist < old_max_dist:
            total_reward += config.SHEEP_HERDING_REWARD  # Reward if the farthest sheep gets closer

        # Small penalty for every step (to encourage faster completion)
        total_reward -= 0.1

        # Penalty if every sheep is not moving (i.e., if no sheep has moved since last step)
        all_sheep_stationary = all(sheep == prev_sheep for sheep, prev_sheep in zip(self.sheep, self.prev_sheep))
        if all_sheep_stationary:
            total_reward -= config.SHEEP_NOT_MOVING_PENALTY  # Add your penalty here

        # Penalty for dogs hitting walls (not moving)
        for i, dog in enumerate(self.dogs):
            if dog == self.prev_dogs[i]:  # If the position didn’t change
                total_reward += config.DOG_HIT_WALL_PENALTY

        # Reward for successfully herding all sheep to the target
        if all(sheep == self.target for sheep in self.sheep):
            total_reward += config.HERD_ALL_SHEEP_REWARD

        # Update previous positions for next step
        self.prev_dogs = [dog[:] for dog in self.dogs]
        self.prev_sheep = [sheep[:] for sheep in self.sheep]

        return total_reward

    def render(self):
        self.renderer.render_game(self.dogs, self.sheep, self.target)

    def close(self):
        self.renderer.close()
