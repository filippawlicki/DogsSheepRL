import sys
import pygame
import torch
import numpy as np
from envs.dogs_sheep_env import DogsSheepEnv
import config
from train_alt import QNetwork as DQN  # Upewnij się, że nazwa klasy odpowiada Twojej implementacji


def decode_action(action_int, num_dogs):
    """
    Dekoduje złożoną akcję (w postaci pojedynczej liczby) na listę akcji dla każdego psa.
    Dla N psów i 4 możliwych ruchów (0: góra, 1: dół, 2: lewo, 3: prawo)
    zakładamy, że złożona akcja zapisana jest w systemie liczbowym o podstawie 4.
    """
    actions = []
    for _ in range(num_dogs):
        actions.append(action_int % 4)
        action_int //= 4
    return actions[::-1]


def process_observation(obs):
    """
    Przekształca obserwację (słownik) w płaski wektor typu float32.
    Wektor zawiera pozycje psów, owiec oraz cel:
      - psy: num_dogs x 2,
      - owce: num_sheep x 2,
      - cel: 2.
    """
    return np.concatenate([
        obs["dogs"].flatten(),
        obs["sheep"].flatten(),
        obs["target"].flatten()
    ]).astype(np.float32)


# Utworzenie środowiska
env = DogsSheepEnv(grid_size=config.GRID_SIZE, num_dogs=config.NUM_DOGS, num_sheep=config.NUM_SHEEP)

# Pobranie początkowej obserwacji i przetworzenie jej na wektor stanu
obs, _ = env.reset()
state = process_observation(obs)

# Obliczenie wymiarów stanu i przestrzeni akcji
state_dim = config.NUM_DOGS * 2 + config.NUM_SHEEP * 2 + 2
action_dim = 4 ** config.NUM_DOGS  # Złożona przestrzeń akcji dla N psów

# Utworzenie modelu i wczytanie stanu wytrenowanego modelu
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/dqn_model_5x5+2d+2o.pth"))
model.eval()


def select_action(model, state):
    """
    Na podstawie bieżącego stanu wybiera akcję używając wytrenowanego modelu.
    Zwraca listę akcji, czyli jeden ruch dla każdego psa.
    """
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        composite_action = int(q_values.argmax(dim=-1).item())
        return decode_action(composite_action, config.NUM_DOGS)


print("Model gra. Naciśnij Q lub zamknij okno, aby zakończyć wizualizację.")
while True:
    env.reset()
    done = False
    while not done:
        # Renderowanie środowiska (odświeża widok)
        env.render()

        # Wybór akcji przez model
        actions = select_action(model, state)

        # Wykonanie ruchu w środowisku
        obs, reward, done, truncated, _ = env.step(actions)
        state = process_observation(obs)

        # Obsługa zdarzeń Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

env.close()