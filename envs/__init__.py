from gym.envs.registration import register

register(
    id="WolfSheep-v0",
    entry_point="envs.wolf_sheep_env:WolfSheepEnv",
)
