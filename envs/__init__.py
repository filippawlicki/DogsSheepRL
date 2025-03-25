from gymnasium.envs.registration import register

register(
    id="DogsSheep-v0",
    entry_point="envs.dogs_sheep_env:DogsSheepEnv",
)
