import sys

import pygame

import config


class GameRenderer:
    def __init__(self, grid_size):
        """Initialize the game screen and load images."""
        pygame.init()
        self.grid_size = grid_size
        self.screen = pygame.display.set_mode((grid_size * config.PYGAME_SCALE,
                                               grid_size * config.PYGAME_SCALE))
        pygame.display.set_caption("Dogs Sheep Simulation")
        self.clock = pygame.time.Clock()  # Create a clock object to control the frame rate

        # Load images
        self.dog_image = pygame.image.load(config.DOG_IMAGE)
        self.sheep_image = pygame.image.load(config.SHEEP_IMAGE)
        self.target_image = pygame.image.load(config.TARGET_IMAGE)

        # Scale images to fit in the grid cells
        self.dog_image = pygame.transform.scale(self.dog_image, (config.PYGAME_SCALE, config.PYGAME_SCALE))
        self.sheep_image = pygame.transform.scale(self.sheep_image, (config.PYGAME_SCALE, config.PYGAME_SCALE))
        self.target_image = pygame.transform.scale(self.target_image, (config.PYGAME_SCALE, config.PYGAME_SCALE))

    def draw_animal(self, animals, image):
        """
        Draw any animal using an image.
        The input `animals` is expected to be a NumPy array of shape (N, 2).
        """
        # Convert numpy array positions into a list of tuples
        positions = [tuple(pos) for pos in animals]
        # Use only unique positions to avoid duplicating images
        unique_positions = set(positions)
        for pos in unique_positions:
            # Using convention: pos[0] is row (y) and pos[1] is column (x)
            x = pos[1] * config.PYGAME_SCALE
            y = pos[0] * config.PYGAME_SCALE
            self.screen.blit(image, (x, y))

            # Count how many animals share this position
            count = positions.count(pos)
            if count > 1:
                font = pygame.font.Font(None, 36)
                # Render the count text in white
                text = font.render(str(count), True, (255, 255, 255))
                # Create a black outline for better readability
                outline = font.render(str(count), True, (0, 0, 0))
                # Draw a simple outline by offsetting the text slightly in four directions
                self.screen.blit(outline, (x - 1, y - 1))
                self.screen.blit(outline, (x + 1, y - 1))
                self.screen.blit(outline, (x - 1, y + 1))
                self.screen.blit(outline, (x + 1, y + 1))
                self.screen.blit(text, (x, y))

    def draw_target(self, target):
        """Draw the target location using its image."""
        # target is assumed to be a NumPy array or list with [row, col]
        x = target[1] * config.PYGAME_SCALE
        y = target[0] * config.PYGAME_SCALE
        self.screen.blit(self.target_image, (x, y))

    def render_game(self, dogs, sheep, target):
        """
        Render the entire game state on the screen.
        Arguments:
          - dogs: NumPy array of dog positions (each a [row, col] pair)
          - sheep: NumPy array of sheep positions (each a [row, col] pair)
          - target: Array-like object for the target position
        """
        self.screen.fill((0, 100, 0))  # Fill the background with green

        # Draw grid lines
        for i in range(self.grid_size):
            pygame.draw.line(
                self.screen, (0, 0, 0),
                (0, i * config.PYGAME_SCALE),
                (self.grid_size * config.PYGAME_SCALE, i * config.PYGAME_SCALE)
            )
            pygame.draw.line(
                self.screen, (0, 0, 0),
                (i * config.PYGAME_SCALE, 0),
                (i * config.PYGAME_SCALE, self.grid_size * config.PYGAME_SCALE)
            )

        # Draw all game elements (dogs, sheep, and target)
        self.draw_animal(dogs, self.dog_image)
        self.draw_animal(sheep, self.sheep_image)
        self.draw_target(target)

        pygame.display.flip()  # Update the display
        self.clock.tick(10)  # Limit the frame rate to 10 FPS

        # Process event queue to handle exit events (e.g., window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()