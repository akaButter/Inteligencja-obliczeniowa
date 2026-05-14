import gymnasium as gym
from stable_baselines3 import PPO
import os
import CatchBrains

env = gym.make("CatchBrains/CatchBrains", render_mode=None, lives = 5)

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_zombie_logs/"
)
model.learn(total_timesteps=200000)

model.save("zombie_agent_v1")
print("Model saved.")