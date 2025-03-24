from pathlib import Path

ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = "output"
GRID_SIZE = 5
PYGAME_SCALE = 80
NUM_DOGS = 2
NUM_SHEEP = 2
TARGET_RADIUS = 2
MIN_DISTANCE_SHEEP = 2
SHEEP_VISION_RANGE = GRID_SIZE*4/5

# Reward settings
# Dog movement rewards
DOG_MOVE_TOWARD_SHEEP_REWARD = 0.5
DOG_MOVE_AWAY_PENALTY = -0.5
DOG_HIT_WALL_PENALTY = -1.0 # Not moving

# Sheep movement rewards
SHEEP_MOVE_TOWARD_TARGET_REWARD = 1.0
SHEEP_CAPTURED_REWARD = 5.0
SHEEP_NOT_MOVING_PENALTY = -0.5

# Herding efficiency
SHEEP_HERDING_REWARD = 2.0
SHEEP_SPREAD_PENALTY = -0.2
DOG_MOVE_REWARD = 0.2

# Final success reward
HERD_ALL_SHEEP_REWARD = 100.0




# Miscellaneous settings
RANDOM_SEED = None

# Path to images
DOG_IMAGE = ROOT_DIR / "images/dog.png"
SHEEP_IMAGE = ROOT_DIR / "images/sheep.png"
TARGET_IMAGE = ROOT_DIR / "images/fence.png"
