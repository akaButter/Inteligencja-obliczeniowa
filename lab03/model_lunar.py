import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 1. Tworzymy środowisko w trybie ciągłym
# 'render_mode' ustawiamy na human, żeby widzieć postępy (ale to spowalnia naukę)
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
env = Monitor(env, log_dir)
# 2. Inicjalizujemy model PPO
# MlpPolicy = Multi-layer Perceptron (zwykła sieć neuronowa)
model = PPO("MlpPolicy", env, verbose=1, device="cpu", learning_rate=0.0003, gamma=0.9) 

# 3. Trenujemy agenta (na początek np. 100 000 kroków)
print("PITU PITU PTIU FIUFIU")
model.learn(total_timesteps=1000)

# 4. Zapisujemy model
model.save("ppo_lunar_lander_continuous")
print("ERROR ERROR NO ERROR MODEL SAVED")

env.close()