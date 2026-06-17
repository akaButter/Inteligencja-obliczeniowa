"""Generowanie wykresów: krzywe uczenia i porównanie win rate."""
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt


def _load_csv_curve(csv_path: str) -> tuple[list, list]:
    steps, rewards = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["timestep"]))
            rewards.append(float(row["ep_reward"]))
    return steps, rewards


def _smooth(values: list, window: int = 50) -> list:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Dopasuj długość przez prepend wartości startowych
    pad = np.full(window - 1, smoothed[0])
    return list(np.concatenate([pad, smoothed]))


def plot_learning_curves(
    configs: list[str],
    results_dir: str = "results",
    out_path: str | None = None,
    window: int = 100,
):
    """Krzywe uczenia (nagroda epizodowa vs kroki) dla podanych konfiguracji."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"ppo_a": "#2196F3", "ppo_b": "#FF5722"}

    for config in configs:
        csv_path = os.path.join(results_dir, config, "learning_curve.csv")
        if not os.path.exists(csv_path):
            print(f"Brak pliku {csv_path}, pomijam.")
            continue
        steps, rewards = _load_csv_curve(csv_path)
        smoothed = _smooth(rewards, window)
        color = colors.get(config, None)
        ax.plot(steps, rewards, alpha=0.2, color=color)
        ax.plot(steps[: len(smoothed)], smoothed, label=config.upper(), color=color, linewidth=2)

    ax.set_xlabel("Kroki (timesteps)")
    ax.set_ylabel("Nagroda epizodowa")
    ax.set_title("Krzywe uczenia – PPO_A vs PPO_B")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(results_dir, "plots", "learning_curves.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Krzywa uczenia: {out_path}")


def plot_win_rates(
    eval_results: dict,
    results_dir: str = "results",
    out_path: str | None = None,
):
    """Wykres słupkowy win rate: jednorodny vs mieszany."""
    configs = [c for c in ("ppo_a", "ppo_b") if f"{c}_all_ppo" in eval_results]
    if not configs:
        print("Brak wyników do wykresu win rate.")
        return

    labels, all_ppo_rates, mixed_ppo_rates = [], [], []
    for c in configs:
        labels.append(c.upper())
        all_ppo = eval_results[f"{c}_all_ppo"]
        mixed = eval_results[f"{c}_mixed"]
        # Średni win rate per agent w scenariuszu jednorodnym
        avg_all = np.mean(list(all_ppo["win_rates"].values()))
        all_ppo_rates.append(avg_all)
        mixed_ppo_rates.append(mixed["ppo_win_rate"])

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, all_ppo_rates, width, label="Jednorodny (PPO vs PPO)", color="#2196F3")
    bars2 = ax.bar(x + width / 2, mixed_ppo_rates, width, label="Mieszany (PPO vs losowy)", color="#FF5722")

    # Baseline losowy – 25% dla 4 graczy
    ax.axhline(0.25, color="gray", linestyle="--", label="Baseline losowy (25%)")

    ax.set_ylabel("Win rate")
    ax.set_title("Porównanie skuteczności agentów")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(results_dir, "plots", "win_rates.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Win rates: {out_path}")


def generate_all_plots(results_dir: str = "results"):
    """Generuje wszystkie wykresy na podstawie zapisanych wyników."""
    plot_learning_curves(["ppo_a", "ppo_b"], results_dir)

    eval_path = os.path.join(results_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
        plot_win_rates(eval_results, results_dir)
    else:
        print(f"Brak {eval_path} – pomiń wykresy win rate.")
