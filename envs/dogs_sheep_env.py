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


        # obs_size = (self.num_dogs + self.num_sheep) * 2 + 2  # (x, y) for each dog, sheep, and target
        # self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(obs_size,), dtype=np.int32)
        self.observation_space = gym.spaces.Dict(
            {
                "dogs": gym.spaces.Box(0, self.grid_size - 1, shape=(self.num_dogs, 2), dtype=int),
                "sheep": gym.spaces.Box(0, self.grid_size - 1, shape=(self.num_sheep, 2), dtype=int),
                "target": gym.spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int)
            }
        )


        # self.action_space = gym.spaces.MultiDiscrete([4] * self.num_dogs)  # Every dog can move in 4 directions
        self.action_space = gym.spaces.Discrete(4 ** config.NUM_DOGS)
        self.renderer = GameRenderer(grid_size)

        # Send target information to Prolog
        self._send_target_to_prolog()
        self._send_grid_size_to_prolog()

        self.steps = 0

        self.state_history = {}  # To track the history of states

    def _send_target_to_prolog(self):
        prolog.retractall("target_position(_, _)")
        prolog.assertz(f"target_position({self.target[0]}, {self.target[1]})")

    def _send_grid_size_to_prolog(self):
        prolog.assertz(f"grid_size({self.grid_size})")

    def _random_pos(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # self.dogs = [self._random_pos() for _ in range(self.num_dogs)]
        # self.sheep = [self._random_pos() for _ in range(self.num_sheep)]
        # self.target = self._random_pos()
        self.dogs = self.np_random.integers(0, self.grid_size, size=(self.num_dogs, 2), dtype=int)
        self.sheep = self.np_random.integers(0, self.grid_size, size=(self.num_sheep, 2), dtype=int)
        self.target = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        # self.prev_dogs = [dog[:] for dog in self.dogs]
        # self.prev_sheep = [sheep[:] for sheep in self.sheep]
        self.prev_dogs = self.dogs.copy()
        self.prev_sheep = self.sheep.copy()

        self._send_target_to_prolog()
        self.steps = 0
        observation = self._get_observation()
        self.state_history.clear()  # Clear state history on reset
        return observation, {}

    def step(self, dog_actions, pushing_sheep=True):
        """
        Makes a step in the environment.
        - Dog actions is a list of actions for each dog.
        - Returns observation, reward, done, truncated, info.
        - Sheep avoid dogs.
        """
        self.steps += 1
        # Store previous sheep positions before movement
        prev_sheep_positions = self.sheep.copy()

        # Compute the total distance BEFORE movement
        old_total_distance = sum(self._distance(s, self.target) for s in self.sheep)
        self._move_dogs(dog_actions, pushing_sheep)
        if not pushing_sheep:
            self._move_sheep()

        # done = self._check_done()
        # truncated = self._check_repeated_state()
        # reward = self._compute_reward()
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
            if distance_delta < 0:
                distance_reward = -2
            else:
                distance_reward = -3
        # Count only sheep that reached the target THIS STEP
        newly_arrived_sheep = sum(
            np.array_equal(s, self.target) and not np.array_equal(prev_sheep_positions[i], self.target)
            for i, s in enumerate(self.sheep)
        )
        sheep_reward = newly_arrived_sheep * 5  # Reward per newly arrived sheep

        sheep_left_target = sum(
            not np.array_equal(s, self.target) and np.array_equal(prev_sheep_positions[i], self.target)
            for i, s in enumerate(self.sheep)
        )
        sheep_penalty = sheep_left_target * -7  # Penalty per sheep that left the target

        total_sheep_reward = sheep_reward + sheep_penalty

        # Large reward if all sheep reach the target
        done = self._check_done()
        goal_reward = 15 + 50 / self.steps if done else 0

        # Small penalty for each step taken by the dogs
        step_penalty = -0.3 * self.num_dogs

        # Total reward
        reward = distance_reward + total_sheep_reward + goal_reward + step_penalty
        #print(f"Distance delta: {distance_delta}")
        #print(f"Distance reward: {distance_reward}")

        return reward, done

    def _check_repeated_state(self):
        state_tuple = tuple(tuple(d) for d in self.dogs) + tuple(tuple(s) for s in self.sheep)

        if state_tuple in self.state_history:
            self.state_history[state_tuple] += 1
        else:
            self.state_history[state_tuple] = 1

        return self.state_history[state_tuple] >= 3  # Truncate if the same state is repeated 3 times

    def _move_dogs(self, actions, pushing_sheep=True):
        directions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        for i, action in enumerate(actions):
            move = directions[action]
            self.dogs[i] = np.clip(self.dogs[i] + move, 0, self.grid_size - 1)

            if pushing_sheep:
                # Convert sheep positions to Prolog format
                sheep_prolog = "[" + ", ".join(f"({sheep[0]}, {sheep[1]})" for sheep in self.sheep) + "]"

                # Prolog query to move sheep if occupied
                query = f"push_sheep({self.dogs[i][0]}, {self.dogs[i][1]}, {move[0]}, {move[1]}, {sheep_prolog}, NewSheepList)"
                # print(f"\nProlog query: {query}")  # Debugging line
                result = list(prolog.query(query))
                # print("result: ", result)

                if result:
                    new_sheep_list = result[0]['NewSheepList']
                    # print("new_sheep_list: ", new_sheep_list)
                    cleaned_sheep_list = [sheep.strip(',()') for sheep in new_sheep_list if sheep.strip(',()')]
                    # print("cleaned_sheep_list: ", cleaned_sheep_list)
                    self.sheep = [list(map(int, sheep.split(','))) for sheep in cleaned_sheep_list]
                    # print("self.sheep: ", self.sheep)


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
        # """Returns a multichannel observation tensor representing the environment."""
        #
        # # Initialize multichannel observation
        # obs = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        #
        # # Channel 1: Dog positions (binary mask)
        # for x, y in self.dogs:
        #     obs[0, x, y] = 1
        #
        #     # Channel 2: Sheep positions (binary mask)
        # for x, y in self.sheep:
        #     obs[1, x, y] = 1
        #
        #     # Channel 3: Target position (binary mask)
        # tx, ty = self.target
        # obs[2, tx, ty] = 1
        #
        # # Channel 4: Distance to nearest sheep for each grid cell
        # for i in range(self.grid_size):
        #     for j in range(self.grid_size):
        #         min_dist = min(np.linalg.norm([i - sx, j - sy]) for sx, sy in self.sheep)
        #         obs[3, i, j] = min_dist / self.grid_size  # Normalize to [0, 1]
        #
        # # Channel 5: Distance to target for each grid cell
        # for i in range(self.grid_size):
        #     for j in range(self.grid_size):
        #         dist = np.linalg.norm([i - tx, j - ty])
        #         obs[4, i, j] = dist / self.grid_size  # Normalize to [0, 1]
        #
        # return obs  # Shape: (5, grid_size, grid_size)

    def _check_done(self):
        return all(np.array_equal(sheep, self.target) for sheep in self.sheep)


    # def _compute_reward(self):
    #     """Compute the reward for the current state."""
    #     total_reward = 0
    #
    #     # Reward for moving dogs, to encourage them to move
    #     for i, dog in enumerate(self.dogs):
    #         if dog != self.prev_dogs[i]:
    #             total_reward += config.DOG_MOVE_REWARD
    #
    #         # Reward for dogs approaching the nearest sheep that is not in the target
    #         nearest_sheep = min(
    #             (sheep for sheep in self.sheep if sheep != self.target),
    #             key=lambda s: self._distance(dog, s),
    #             default=None
    #         )
    #         if nearest_sheep:
    #             old_distance = self._distance(self.prev_dogs[i], nearest_sheep)
    #             new_distance = self._distance(dog, nearest_sheep)
    #             if new_distance < old_distance:
    #                 total_reward += config.DOG_MOVE_TOWARD_SHEEP_REWARD
    #             else:
    #                 total_reward += config.DOG_MOVE_AWAY_PENALTY
    #
    #     # Reward for sheep moving toward target
    #     for i, sheep in enumerate(self.sheep):
    #         old_distance = self._distance(self.prev_sheep[i], self.target)
    #         new_distance = self._distance(sheep, self.target)
    #
    #         if new_distance < old_distance:
    #             total_reward += config.SHEEP_MOVE_TOWARD_TARGET_REWARD
    #         else:
    #             total_reward += config.SHEEP_SPREAD_PENALTY
    #
    #         if new_distance < 1 and new_distance < old_distance:
    #             total_reward += config.SHEEP_CAPTURED_REWARD
    #
    #     # Reward for reducing the max sheep distance from the target
    #     old_max_dist = max(self._distance(s, self.target) for s in self.prev_sheep)
    #     new_max_dist = max(self._distance(s, self.target) for s in self.sheep)
    #
    #     if new_max_dist < old_max_dist:
    #         total_reward += config.SHEEP_HERDING_REWARD  # Reward if the farthest sheep gets closer
    #
    #     if new_max_dist > old_max_dist:
    #         total_reward += config.SHEEP_SPREAD_PENALTY  # Penalty if the farthest sheep gets farther
    #
    #     # Small penalty for every step (to encourage faster completion)
    #     total_reward -= 0.1
    #
    #     # Penalty if every sheep is not moving (i.e., if no sheep has moved since last step)
    #     all_sheep_stationary = all(sheep == prev_sheep for sheep, prev_sheep in zip(self.sheep, self.prev_sheep))
    #     if all_sheep_stationary:
    #         total_reward += config.SHEEP_NOT_MOVING_PENALTY  # Add your penalty here
    #
    #     # Penalty for dogs hitting walls (not moving)
    #     for i, dog in enumerate(self.dogs):
    #         if dog == self.prev_dogs[i]:  # If the position didn’t change
    #             total_reward += config.DOG_HIT_WALL_PENALTY
    #
    #     # Reward for successfully herding all sheep to the target
    #     if all(sheep == self.target for sheep in self.sheep):
    #         total_reward += config.HERD_ALL_SHEEP_REWARD
    #
    #     # Update previous positions for next step
    #     self.prev_dogs = [dog[:] for dog in self.dogs]
    #     self.prev_sheep = [sheep[:] for sheep in self.sheep]
    #
    #     return total_reward

    def render(self):
        self.renderer.render_game(self.dogs, self.sheep, self.target)

    def close(self):
        self.renderer.close()
