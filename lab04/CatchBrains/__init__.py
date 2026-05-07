from gymnasium.envs.registration import register

register(
    id="CatchBrains/CatchBrains",
    entry_point="CatchBrains.envs:CatchBrainsEnv",
    max_episode_steps=500,
)
