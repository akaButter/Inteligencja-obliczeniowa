import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
        return True


SCRIPT_DIR = os.path.dirname(__file__) or "."
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "lunar_dqn")


def train_dqn(total_timesteps=80_000, gamma=0.9, learning_rate=6.3e-4,
              save_dir=DEFAULT_RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")
    env = Monitor(env, save_dir)

    model = DQN(
        "MlpPolicy", env,
        verbose=0,
        device="cpu",
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        target_update_interval=500,
        train_freq=4,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
    )

    print(f"[DQN-SB3] Trening Lunar Lander (discrete): "
          f"timesteps={total_timesteps}, gamma={gamma}, lr={learning_rate}")
    cb = RewardLoggerCallback()
    try:
        model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True)
    except ImportError:
        model.learn(total_timesteps=total_timesteps, callback=cb)

    model.save(os.path.join(save_dir, "model"))
    env.close()

    rewards, lengths = cb.episode_rewards, cb.episode_lengths

    with open(os.path.join(save_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "length"])
        for i, (r, l) in enumerate(zip(rewards, lengths)):
            w.writerow([i, r, l])

    summary = compute_summary(rewards, lengths, gamma)
    summary["algorithm"] = "DQN (SB3)"
    summary["total_timesteps"] = total_timesteps
    summary["learning_rate"] = learning_rate
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_curve(rewards, lengths,
               title=f"DQN (Lunar Lander discrete) γ={gamma}, lr={learning_rate}",
               out_path=os.path.join(save_dir, "curve.png"))
    print_summary(summary)
    return rewards, lengths, summary, model


def compute_summary(rewards, lengths, gamma):
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    n = len(rewards)
    if n == 0:
        return {"gamma": gamma, "n_episodes": 0}
    last_q = rewards[3 * n // 4:]
    first_q = rewards[: max(1, n // 4)]
    return {
        "gamma": gamma,
        "n_episodes": int(n),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_first_25%": float(np.mean(first_q)),
        "mean_last_25%": float(np.mean(last_q)),
        "improvement": float(np.mean(last_q) - np.mean(first_q)),
        "mean_length": float(np.mean(lengths)),
        "solved_episodes": int(np.sum(rewards > 200)),
        "solved_pct": float(100 * np.sum(rewards > 200) / n),
        "discounted_return_first_1000": float(
            sum(r * (gamma ** i) for i, r in enumerate(rewards[:1000]))
        ),
    }


def print_summary(s):
    print("\n" + "=" * 60)
    print(f" PODSUMOWANIE: {s.get('algorithm','')} (γ={s.get('gamma')})")
    print("=" * 60)
    if s.get("n_episodes", 0) == 0:
        print("  Brak ukończonych epizodów.")
        return
    print(f"  Liczba ukończonych epizodów:    {s['n_episodes']}")
    print(f"  Najlepszy / Najgorszy:          {s['best_reward']:.1f} / {s['worst_reward']:.1f}")
    print(f"  Średnia (cały trening):         {s['mean_reward']:.1f} ± {s['std_reward']:.1f}")
    print(f"  Średnia (pierwsze 25%):         {s['mean_first_25%']:.1f}")
    print(f"  Średnia (ostatnie 25%):         {s['mean_last_25%']:.1f}")
    print(f"  Poprawa (last - first):         {s['improvement']:+.1f}")
    print(f"  Średnia długość epizodu:        {s['mean_length']:.0f} kroków")
    print(f"  Udane lądowania (>200):         {s['solved_episodes']} ({s['solved_pct']:.1f}%)")
    print(f"  Zdyskontowana suma (1000 ep):   {s['discounted_return_first_1000']:.1f}")
    if s["mean_last_25%"] > 200:
        verdict = "Agent NAUCZYŁ SIĘ rozwiązywać zadanie."
    elif s["improvement"] > 50:
        verdict = "Wyraźna progresja - dłuższy trening prawdopodobnie da rozwiązanie."
    elif s["improvement"] > 0:
        verdict = "Powolne uczenie."
    else:
        verdict = "Brak postępów - zmień hiperparametry lub trenuj dłużej."
    print(f"\n  Wniosek: {verdict}")
    print("=" * 60 + "\n")


def plot_curve(rewards, lengths, title, out_path):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    episodes = np.arange(len(rewards))
    window = max(10, len(rewards) // 20) if len(rewards) > 10 else 1

    axes[0].plot(episodes, rewards, alpha=0.3, color="lightcoral", label="surowa")
    if len(rewards) >= window > 1:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window - 1, len(rewards)), ma,
                     color="darkred", linewidth=2, label=f"śr. krocząca ({window})")
    axes[0].axhline(200, color="green", linestyle="--", alpha=0.6, label="próg sukcesu (200)")
    axes[0].axhline(0, color="black", linestyle="-", alpha=0.2)
    axes[0].set_xlabel("Epizod"); axes[0].set_ylabel("Suma nagród")
    axes[0].set_title("Krzywa uczenia"); axes[0].grid(True, alpha=0.4); axes[0].legend()

    axes[1].plot(episodes, lengths, alpha=0.3, color="khaki", label="surowa")
    if len(lengths) >= window > 1:
        ma_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
        axes[1].plot(np.arange(window - 1, len(lengths)), ma_l,
                     color="olive", linewidth=2, label=f"śr. krocząca ({window})")
    axes[1].set_xlabel("Epizod"); axes[1].set_ylabel("Liczba kroków")
    axes[1].set_title("Długość epizodów"); axes[1].grid(True, alpha=0.4); axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    train_dqn(total_timesteps=80_000, gamma=0.9)
