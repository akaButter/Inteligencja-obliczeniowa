import os
import json
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


SCRIPT_DIR = os.path.dirname(__file__) or "."
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "lunar_experiments")


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
        return True


def run_one(gamma, learning_rate=3e-4, total_timesteps=30_000, seed=0):
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=0, device="cpu",
                learning_rate=learning_rate, gamma=gamma, seed=seed,
                n_steps=2048, batch_size=64)
    cb = RewardLoggerCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    env.close()
    return cb.episode_rewards, cb.episode_lengths


def discounted_total(rewards, gamma, n_first=1000):
    """Całkowita zdyskontowana nagroda w pierwszych n_first epizodach."""
    return float(sum(r * (gamma ** i) for i, r in enumerate(rewards[:n_first])))


# Porównanie 3 współczynników dyskontowych
def gamma_comparison(save_dir=DEFAULT_RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    gammas = [0.5, 0.9, 0.99]
    results = {}
    for g in gammas:
        print(f"\n--- gamma = {g} ---")
        rewards, lengths = run_one(gamma=g, total_timesteps=30_000)
        results[g] = {
            "rewards": rewards,
            "lengths": lengths,
            "mean": float(np.mean(rewards)) if rewards else 0.0,
            "mean_last_25%": float(np.mean(rewards[3 * len(rewards) // 4:])) if rewards else 0.0,
            "best": float(np.max(rewards)) if rewards else 0.0,
            "discounted_first_1000": discounted_total(rewards, g, 1000),
            "n_episodes": len(rewards),
        }

    # Wykres porównawczy
    plt.figure(figsize=(11, 6))
    colors = {0.5: "tab:blue", 0.9: "tab:green", 0.99: "tab:red"}
    for g, data in results.items():
        rew = np.array(data["rewards"])
        if len(rew) == 0:
            continue
        window = max(10, len(rew) // 20)
        ma = np.convolve(rew, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, len(rew)), ma,
                 color=colors[g], linewidth=2, label=f"γ = {g}")
    plt.axhline(200, color="grey", linestyle="--", alpha=0.5, label="próg sukcesu")
    plt.axhline(0, color="black", linestyle="-", alpha=0.2)
    plt.xlabel("Epizod"); plt.ylabel("Nagroda (śr. krocząca)")
    plt.title("Wpływ współczynnika dyskontowego γ - PPO, Lunar Lander (30k kroków)")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    out_png = os.path.join(save_dir, "gamma_comparison.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()

    # Zapis CSV
    with open(os.path.join(save_dir, "gamma_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "n_episodes", "mean", "mean_last_25%", "best", "discounted_first_1000"])
        for g, d in results.items():
            w.writerow([g, d["n_episodes"], f"{d['mean']:.2f}",
                        f"{d['mean_last_25%']:.2f}", f"{d['best']:.2f}",
                        f"{d['discounted_first_1000']:.2f}"])

    summary = {str(g): {k: v for k, v in d.items() if k not in ("rewards", "lengths")}
               for g, d in results.items()}
    with open(os.path.join(save_dir, "gamma_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(" PODSUMOWANIE: PORÓWNANIE WSPÓŁCZYNNIKÓW γ")
    print("=" * 60)
    print(f"  {'γ':>6} {'mean':>10} {'last 25%':>10} {'best':>10} {'disc.1000':>12}")
    for g, d in results.items():
        print(f"  {g:>6} {d['mean']:>10.1f} {d['mean_last_25%']:>10.1f} "
              f"{d['best']:>10.1f} {d['discounted_first_1000']:>12.1f}")
    best_g = max(results.items(), key=lambda x: x[1]["mean_last_25%"])[0]
    print(f"\n  Najlepsza γ wg średniej z ostatnich 25% epiz.: {best_g}")
    print("=" * 60 + "\n")
    return results


# Optymalizacja hiperparametrów (grid search)
def hyperparam_search(save_dir=DEFAULT_RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    """Mały grid: 2 lr x 2 gamma. Cel: maksymalizacja zdyskontowanej nagrody w
    pierwszych 1000 epizodach (jak w treści zadania na 8 pkt)."""
    grid = {
        "learning_rate": [1e-4, 3e-4],
        "gamma": [0.9, 0.99],
    }
    results = []
    for lr, g in itertools.product(grid["learning_rate"], grid["gamma"]):
        print(f"\n--- HPO: lr={lr}, gamma={g} ---")
        rewards, lengths = run_one(gamma=g, learning_rate=lr, total_timesteps=20_000)
        score = discounted_total(rewards, g, 1000)
        last25 = float(np.mean(rewards[3 * len(rewards) // 4:])) if rewards else 0.0
        results.append({
            "learning_rate": lr,
            "gamma": g,
            "n_episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_last_25%": last25,
            "discounted_first_1000": score,
        })

    # Sort od najlepszego
    results.sort(key=lambda r: r["discounted_first_1000"], reverse=True)

    print("\n" + "=" * 70)
    print(" PODSUMOWANIE: OPTYMALIZACJA HIPERPARAMETRÓW (PPO Lunar Lander)")
    print(" Cel: max zdyskontowana nagroda w pierwszych 1000 epizodach")
    print("=" * 70)
    print(f"  {'lr':>10} {'γ':>8} {'mean':>10} {'last 25%':>10} {'disc.1000':>12}")
    for r in results:
        marker = " <-- najlepszy" if r is results[0] else ""
        print(f"  {r['learning_rate']:>10.4f} {r['gamma']:>8} "
              f"{r['mean_reward']:>10.1f} {r['mean_last_25%']:>10.1f} "
              f"{r['discounted_first_1000']:>12.1f}{marker}")
    print("=" * 70 + "\n")

    # Heatmapa
    lrs = sorted(set(r["learning_rate"] for r in results))
    gms = sorted(set(r["gamma"] for r in results))
    grid_vals = np.zeros((len(lrs), len(gms)))
    for r in results:
        i, j = lrs.index(r["learning_rate"]), gms.index(r["gamma"])
        grid_vals[i, j] = r["discounted_first_1000"]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(grid_vals, cmap="viridis", aspect="auto")
    plt.xticks(range(len(gms)), [f"γ={g}" for g in gms])
    plt.yticks(range(len(lrs)), [f"lr={lr}" for lr in lrs])
    for i in range(len(lrs)):
        for j in range(len(gms)):
            plt.text(j, i, f"{grid_vals[i, j]:.0f}",
                     ha="center", va="center", color="white", fontsize=11)
    plt.colorbar(im, label="zdyskontowana nagroda (1000 epiz.)")
    plt.title("HPO: zdyskontowana suma nagród (PPO Lunar Lander)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hpo_heatmap.png"),
                dpi=120, bbox_inches="tight")
    plt.show()

    with open(os.path.join(save_dir, "hpo_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    gamma_comparison()
    hyperparam_search()
