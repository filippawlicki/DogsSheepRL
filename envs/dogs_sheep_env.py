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
        self.dogs = np.array([[-1, -1]] * self.num_dogs, dtype=np.int32)
        self.sheep = np.array([[-1, -1]] * self.num_sheep, dtype=np.int32)
        self.target = np.array([-1, -1], dtype=np.int32)
        self.prev_dogs = self.dogs.copy()
        self.prev_sheep = self.sheep.copy()

        self.observation_space = gym.spaces.Dict(
            {
                "dogs": gym.spaces.Box(0, self.grid_size - 1, shape=(self.num_dogs, 2), dtype=int),
                "sheep": gym.spaces.Box(0, self.grid_size - 1, shape=(self.num_sheep, 2), dtype=int),
                "target": gym.spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int)
            }
        )

        # Original action_space here; adjust if necessary for composite actions.
        self.action_space = gym.spaces.Discrete(4 ** config.NUM_DOGS)
        self.renderer = GameRenderer(grid_size)

        # Send target information to Prolog
        self._send_target_to_prolog()

        self.steps = 0

    def _send_target_to_prolog(self):
        prolog.retractall("target_position(_, _)")
        prolog.assertz(f"target_position({self.target[0]}, {self.target[1]})")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dogs = self.np_random.integers(0, self.grid_size, size=(self.num_dogs, 2), dtype=int)
        self.sheep = self.np_random.integers(0, self.grid_size, size=(self.num_sheep, 2), dtype=int)
        self.target = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        self._send_target_to_prolog()
        self.steps = 0
        observation = self._get_observation()
        return observation, {}

    def step(self, dog_actions):
        """
        Makes a step in the environment.
          - dog_actions: a list of actions for each dog.
          - Returns: observation, reward, done, truncated, info.
          - Sheep avoid dogs.
        """
        self.steps += 1
        # Store previous sheep positions before movement
        prev_sheep_positions = self.sheep.copy()

        # Compute the total distance BEFORE movement
        old_total_distance = sum(self._distance(s, self.target) for s in self.sheep)
        #print("Dog actions: ", dog_actions)
        self._move_dogs(dog_actions)
        self._move_sheep()

        observation = self._get_observation()

        # Compute reward using previous sheep positions
        reward, done = self._compute_reward(old_total_distance, prev_sheep_positions)
        truncated = False  # Not used in this environment

        return observation, reward, done, truncated, {}

    def _compute_reward(self, old_total_distance, prev_sheep_positions):
        """
        Compute the reward based on:
          - Reduction in total sheep distance to target.
          - Rewarding only newly arrived sheep at the target.
          - Penalizing movement away from the target.
        """
        # Consider only sheep that are NOT at the target
        moving_sheep = [s for s in self.sheep if not np.array_equal(s, self.target)]

        # Compute total distance only for moving sheep
        new_total_distance = sum(self._distance(s, self.target) for s in moving_sheep)

        # Reward for reducing distance (only for moving sheep)
        distance_delta = old_total_distance - new_total_distance  # Positive if sheep moved closer

        if distance_delta > 0:
            distance_reward = 1
        else:
            if distance_delta == 0:
                distance_reward = -3
            else:
                distance_reward = -2
        # Count only sheep that reached the target THIS STEP
        newly_arrived_sheep = sum(
            np.array_equal(s, self.target) and not np.array_equal(prev_sheep_positions[i], self.target)
            for i, s in enumerate(self.sheep)
        )
        sheep_reward = newly_arrived_sheep * 5  # Reward per newly arrived sheep

        # Large reward if all sheep reach the target
        done = self._check_done()
        goal_reward = 15 + 50 / self.steps if done else 0

        # Total reward
        reward = distance_reward + sheep_reward + goal_reward
        #print(f"Distance delta: {distance_delta}")
        #print(f"Distance reward: {distance_reward}")

        return reward, done

    def _move_dogs(self, actions):
        directions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        for i, action in enumerate(actions):
            move = directions[action]
            self.dogs[i] = np.clip(self.dogs[i] + move, 0, self.grid_size - 1)

    def _move_sheep(self):
        """Sheep move randomly, avoiding dogs and each other."""
        for i in range(self.num_sheep):
            # Skip sheep that are already at the target
            if np.array_equal(self.sheep[i], self.target):
                continue

            # Calculate crowdedness using distance (can be modified as needed)
            crowded_sheep = sum(1 for s in self.sheep if self._distance(s, self.sheep[i]) < config.MIN_DISTANCE_SHEEP)

            dogs_prolog = "[" + ", ".join(f"({dog[0]}, {dog[1]})" for dog in self.dogs) + "]"
            sheep_prolog = "[" + ", ".join(f"({sheep[0]}, {sheep[1]})" for sheep in self.sheep) + "]"

            query = f"direction_to_move({self.sheep[i][0]}, {self.sheep[i][1]}, {dogs_prolog}, {crowded_sheep}, {config.SHEEP_VISION_RANGE}, {sheep_prolog}, MoveX, MoveY)"
            result = list(prolog.query(query))

            if not result:
                move_x, move_y = 0, 0
            else:
                try:
                    move_x = result[0]['MoveX']
                    move_y = result[0]['MoveY']
                except (ValueError, KeyError) as e:
                    print(f"Error converting Prolog result to int: {e}")
                    move_x, move_y = 0, 0

            self.sheep[i][0] = np.clip(self.sheep[i][0] + move_x, 0, self.grid_size - 1)
            self.sheep[i][1] = np.clip(self.sheep[i][1] + move_y, 0, self.grid_size - 1)

    def _distance(self, pos1, pos2):
        # Uses Euclidean distance between two points.
        return np.linalg.norm(pos1 - pos2)

    def _get_observation(self):
        return {"dogs": self.dogs, "sheep": self.sheep, "target": self.target}

    def _check_done(self):
        return all(np.array_equal(sheep, self.target) for sheep in self.sheep)


    def render(self):
        self.renderer.render_game(self.dogs, self.sheep, self.target)

    def close(self):
        self.renderer.close()