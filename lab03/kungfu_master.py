import gymnasium as gym
import ale_py

# Ręczna rejestracja środowisk Atari (wymagane w Gymnasium >= 1.0.0)
gym.register_envs(ale_py)

# Initialise the environment
# Action Space - Discrete(14)
# Observation Space - Box(0, 255, (210, 160, 3), uint8)
env = gym.make("ALE/KungFuMaster-v5")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()