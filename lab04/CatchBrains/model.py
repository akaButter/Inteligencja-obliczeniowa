import gymnasium as gym
from stable_baselines3 import PPO
import os
import CatchBrains
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import matplotlib.pyplot as plt


env = gym.make("CatchBrains/CatchBrains", render_mode=None, lives = 5)

env = Monitor(env, "./logs/")


model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_zombie_logs/"
)
model.learn(total_timesteps=400000,
            progress_bar=True)


df = pd.read_csv("./logs/monitor.csv", skiprows=1)

plt.plot(df.index, df['r'])
plt.title("Suma nagród w kolejnych epizodach")
plt.xlabel("Epizod")
plt.ylabel("Total Reward")
plt.show()

model.save("zombie_agent_v2")
print("Model saved.")