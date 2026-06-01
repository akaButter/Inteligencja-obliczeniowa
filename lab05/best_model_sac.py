import json
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

ENV_NAME = "LunarLanderContinuous-v3"
env = gym.make(ENV_NAME)

eval_env = gym.make(ENV_NAME)

params = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_size": 300000,
    "train_freq": 1,
    "gradient_steps": 1,
}

policy_kwargs = {"net_arch": [256, 256]}

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    tau=params["tau"],
    batch_size=params["batch_size"],
    buffer_size=params["buffer_size"],
    train_freq=params["train_freq"],
    gradient_steps=params["gradient_steps"],
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./sac_lunar_tensorboard/",
)

TOTAL_TIMESTEPS = 300000
print(f"Rozpoczynanie uczenia przez {TOTAL_TIMESTEPS} kroków...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

MODEL_ZIP_PATH = "best_model_sac.zip"
model.save(MODEL_ZIP_PATH)
print(f"Model został zapisany do pliku: {MODEL_ZIP_PATH}")

log_results = np.load("./logs/evaluations.npz")
timesteps = log_results["timesteps"].tolist()
results = log_results["results"]

mean_rewards = np.mean(results, axis=1).tolist()
std_rewards = np.std(results, axis=1).tolist()

curve_data = {"timesteps": timesteps, "mean": mean_rewards, "std": std_rewards}

JSON_PATH = "curve_sac.json"
with open(JSON_PATH, "w") as f:
    json.dump(curve_data, f)

print(f"Krzywa uczenia została zapisana do pliku: {JSON_PATH}")

env.close()
eval_env.close()