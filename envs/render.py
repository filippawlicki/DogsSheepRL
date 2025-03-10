import sys

import pygame

class GameRenderer:
    def __init__(self, grid_size):
        """Initialize the game screen."""
        pygame.init()
        self.grid_size = grid_size
        self.screen = pygame.display.set_mode((grid_size * 40, grid_size * 40))
        pygame.display.set_caption("Wolf Sheep Simulation")
        self.clock = pygame.time.Clock()  # Create clock object to control frame rate

    def draw_wolves(self, wolves):
        """Draw wolves on the board."""
        for w in wolves:
            pygame.draw.circle(self.screen, (255, 0, 0), (w[1] * 40 + 20, w[0] * 40 + 20), 10)  # Red for wolves

    def draw_sheep(self, sheep):
        """Draw sheep on the board."""
        for s in sheep:
            pygame.draw.circle(self.screen, (0, 255, 0), (s[1] * 40 + 20, s[0] * 40 + 20), 10)  # Green for sheep

    def draw_target(self, target):
        """Draw the target on the board."""
        pygame.draw.rect(self.screen, (0, 0, 255), (target[1] * 40, target[0] * 40, 40, 40))  # Blue for the target

    def render_game(self, wolves, sheep, target):
        """Render the entire game state on the screen."""
        self.screen.fill((255, 255, 255))  # Fill the background with white

        self.draw_wolves(wolves)
        self.draw_sheep(sheep)
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
