from pathlib import Path

ROOT_DIR = Path(__file__).parent
GRID_SIZE = 10
PYGAME_SCALE = 80
NUM_DOGS = 3
NUM_SHEEP = 5
TARGET_RADIUS = 2
MIN_DISTANCE_SHEEP = 2
SHEEP_VISION_RANGE = GRID_SIZE*4/5

# Reward settings
SHEEP_NEAR_TARGET_REWARD = 5
SHEEP_CAPTURED_PENALTY = -3
SHEEP_MOVEMENT_PENALTY = -1
DOG_MOVEMENT_REWARD = 1

# Miscellaneous settings
RANDOM_SEED = None

# DQN Hyperparameters
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 4  # Up, Down, Left, Right
HIDDEN_SIZE = 128

# Training Parameters
BATCH_SIZE = 32  # Number of experiences sampled from replay buffer
GAMMA = 0.99  # Discount factor for future rewards
LEARNING_RATE = 0.001  # Learning rate for optimizer
MEMORY_SIZE = 10000  # Maximum size of replay buffer
TARGET_UPDATE = 10  # Number of episodes before updating target network
EPSILON = 1.0  # Initial epsilon for epsilon-greedy policy
EPSILON_MIN = 0.01  # Minimum epsilon
EPSILON_DECAY = 0.995  # Epsilon decay rate

# Training Control
NUM_EPISODES = 1000  # Total number of training episodes
MAX_STEPS_PER_EPISODE = 200  # Max steps per episode

# Path to images
DOG_IMAGE = ROOT_DIR / "images/dog.png"
SHEEP_IMAGE = ROOT_DIR / "images/sheep.png"
TARGET_IMAGE = ROOT_DIR / "images/fence.png"
