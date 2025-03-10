import sys
import pygame
import config

class GameRenderer:
    def __init__(self, grid_size):
        """Initialize the game screen and load images."""
        pygame.init()
        self.grid_size = grid_size
        self.screen = pygame.display.set_mode((grid_size * config.PYGAME_SCALE, grid_size * config.PYGAME_SCALE))
        pygame.display.set_caption("Wolf Sheep Simulation")
        self.clock = pygame.time.Clock()  # Create clock object to control frame rate

        # Load images
        self.wolf_image = pygame.image.load(config.WOLF_IMAGE)
        self.sheep_image = pygame.image.load(config.SHEEP_IMAGE)
        self.target_image = pygame.image.load(config.TARGET_IMAGE)

        # Scale images to fit the grid cells
        self.wolf_image = pygame.transform.scale(self.wolf_image, (config.PYGAME_SCALE, config.PYGAME_SCALE))
        self.sheep_image = pygame.transform.scale(self.sheep_image, (config.PYGAME_SCALE, config.PYGAME_SCALE))
        self.target_image = pygame.transform.scale(self.target_image, (config.PYGAME_SCALE, config.PYGAME_SCALE))

    def draw_animal(self, animals, image):
        """Draw any animal using an image."""
        for a in animals:
            x, y = a[1] * config.PYGAME_SCALE, a[0] * config.PYGAME_SCALE  # Position based on grid coordinates
            self.screen.blit(image, (x, y))

            # Check if more than one animal is in the same cell
            count = sum(
                1 for animal in animals if animal == a)  # Count how many times the same animal appears in the list
            if count > 1:
                # Draw a number to indicate the number of animals in the cell
                font = pygame.font.Font(None, 36)

                # Render the text in white
                text = font.render(str(count), True, (255, 255, 255))

                # Create the outline by rendering the text in black and shifting it
                outline = font.render(str(count), True, (0, 0, 0))

                # Draw the outline first (slightly offset in different directions)
                self.screen.blit(outline, (x - 1, y - 1))  # Top-left
                self.screen.blit(outline, (x + 1, y - 1))  # Top-right
                self.screen.blit(outline, (x - 1, y + 1))  # Bottom-left
                self.screen.blit(outline, (x + 1, y + 1))  # Bottom-right
                self.screen.blit(text, (x, y))

    def draw_target(self, target):
        """Draw the target using an image."""
        x, y = target[1] * config.PYGAME_SCALE, target[0] * config.PYGAME_SCALE  # Position based on grid coordinates
        self.screen.blit(self.target_image, (x, y))

    def render_game(self, wolves, sheep, target):
        """Render the entire game state on the screen."""
        self.screen.fill((0, 100, 0))  # Fill the background with green
        # Grid lines
        for i in range(self.grid_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * config.PYGAME_SCALE), (self.grid_size * config.PYGAME_SCALE, i * config.PYGAME_SCALE))
            pygame.draw.line(self.screen, (0, 0, 0), (i * config.PYGAME_SCALE, 0), (i * config.PYGAME_SCALE, self.grid_size * config.PYGAME_SCALE))


        # Draw all game elements (wolves, sheep, and target)
        self.draw_animal(wolves, self.wolf_image)
        self.draw_animal(sheep, self.sheep_image)
        self.draw_target(target)

        pygame.display.flip()  # Update the screen with the drawn elements
        self.clock.tick(10)  # Limit frame rate

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()
