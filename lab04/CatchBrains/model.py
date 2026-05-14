import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import numpy as np
import CatchBrains

MAX_EP_STEPS = 4000  # enough to win (10 brains × ~120 spawn interval)


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + progress_remaining * (start - end)
    return schedule


def make_env(log_dir, seed=0):
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make("CatchBrains/CatchBrains", render_mode=None, lives=5)
    env = TimeLimit(env, max_episode_steps=MAX_EP_STEPS)
    env = Monitor(env, log_dir)
    env.reset(seed=seed)
    return env


train_env = make_env("./logs/", seed=42)
eval_env  = make_env("./logs_eval/", seed=99)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./",
    log_path="./logs_eval/",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./ppo_zombie_logs/",
    # schedules: aggressive early, conservative late → prevents late-training regression
    learning_rate=linear_schedule(3e-4, 3e-5),
    clip_range=linear_schedule(0.2, 0.05),
    ent_coef=0.005,
    n_steps=4096,
    batch_size=128,
    n_epochs=4,
    gamma=0.995,
    gae_lambda=0.95,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)

model.learn(
    total_timesteps=500_000,
    progress_bar=True,
    callback=eval_callback,
)

# save final model, but best_model.zip (from EvalCallback) is usually better
model.save("zombie_agent_v3")
print("Model saved. Najlepszy model: best_model.zip")

# ── learning curve from EvalCallback (100 evenly-spaced eval points) ─────────
eval_data    = np.load("./logs_eval/evaluations.npz")
timesteps    = eval_data["timesteps"]
mean_rewards = eval_data["results"].mean(axis=1)
std_rewards  = eval_data["results"].std(axis=1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(timesteps,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.25, color="steelblue", label="±1 std")
ax.plot(timesteps, mean_rewards, color="steelblue", linewidth=2, label="Średnia nagroda")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Krzywa uczenia agenta (ewaluacja co 10 000 kroków, 20 epizodów)")
ax.set_xlabel("Kroki treningu")
ax.set_ylabel("Średnia nagroda")
ax.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()
print(f"Najlepsza średnia nagroda: {mean_rewards.max():.1f} @ krok {timesteps[mean_rewards.argmax()]}")
