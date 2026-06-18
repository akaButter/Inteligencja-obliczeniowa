"""Generowanie wykresów: krzywe uczenia, win rate, model vs model."""
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

ALGO_COLORS = {"ppo_a": "#2196F3", "ppo_b": "#FF5722", "trpo": "#4CAF50"}
ALGO_LABELS = {"ppo_a": "PPO_A", "ppo_b": "PPO_B", "trpo": "TRPO"}


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

    for config in configs:
        csv_path = os.path.join(results_dir, config, "learning_curve.csv")
        if not os.path.exists(csv_path):
            print(f"Brak pliku {csv_path}, pomijam.")
            continue
        steps, rewards = _load_csv_curve(csv_path)
        smoothed = _smooth(rewards, window)
        color = ALGO_COLORS.get(config)
        label = ALGO_LABELS.get(config, config.upper())
        ax.plot(steps, rewards, alpha=0.15, color=color)
        ax.plot(steps[: len(smoothed)], smoothed, label=label, color=color, linewidth=2)

    ax.set_xlabel("Kroki (timesteps)")
    ax.set_ylabel("Nagroda epizodowa")
    ax.set_title("Krzywe uczenia – PPO_A vs PPO_B vs TRPO")
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
    """Wykres słupkowy: jednorodny vs model_vs_losowy vs baseline dla każdego algo."""
    configs = [c for c in ("ppo_a", "ppo_b", "trpo") if f"{c}_jednorodny" in eval_results]
    if not configs:
        print("Brak wyników do wykresu win rate.")
        return

    labels, jednorodny_rates, vs_random_rates = [], [], []
    baseline_avg = np.mean(list(eval_results.get("baseline_random", {}).get("win_rates", {0: 0.25}).values()))

    for c in configs:
        labels.append(ALGO_LABELS.get(c, c.upper()))
        j = eval_results[f"{c}_jednorodny"]
        avg_j = np.mean(list(j["win_rates"].values()))
        jednorodny_rates.append(avg_j)

        vs = eval_results.get(f"{c}_vs_random", {})
        vs_random_rates.append(vs.get("model_win_rate", 0.0))

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ALGO_COLORS.get(c, "#888") for c in configs]

    b1 = ax.bar(x - width, jednorodny_rates, width, label="Jednorodny (avg/agent)", color=colors, alpha=0.9)
    b2 = ax.bar(x, vs_random_rates, width, label="Model vs losowy (2v2 łącznie)", color=colors, alpha=0.55)
    ax.axhline(baseline_avg, color="gray", linestyle="--", linewidth=1.2, label=f"Baseline losowy ({baseline_avg:.1%})")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="50% (par. dla 2v2)")

    ax.set_ylabel("Win rate")
    ax.set_title("Skuteczność algorytmów – jednorodny i vs losowy")
    ax.set_xticks(x - width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(results_dir, "plots", "win_rates.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Win rates: {out_path}")


def plot_model_vs_model(
    eval_results: dict,
    results_dir: str = "results",
    out_path: str | None = None,
):
    """Wykres head-to-head: wyniki konfrontacji model vs model."""
    matchups = [(k, v) for k, v in eval_results.items() if "team_a_win_rate" in v]
    if not matchups:
        print("Brak wyników model vs model.")
        return

    labels, rates_a, rates_b = [], [], []
    for key, res in matchups:
        ca = res.get("config_a", "?")
        cb = res.get("config_b", "?")
        labels.append(f"{ALGO_LABELS.get(ca, ca)}\nvs\n{ALGO_LABELS.get(cb, cb)}")
        rates_a.append(res["team_a_win_rate"])
        rates_b.append(res["team_b_win_rate"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2.5), 5))
    ax.bar(x - width / 2, rates_a, width, label="Drużyna A (player_0+1)", color="#5C6BC0")
    ax.bar(x + width / 2, rates_b, width, label="Drużyna B (player_2+3)", color="#EF5350")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50%")

    ax.set_ylabel("Łączny win rate drużyny")
    ax.set_title("Model vs Model – konfrontacje bezpośrednie")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bars, rates in [(ax.containers[0], rates_a), (ax.containers[1], rates_b)]:
        for bar, r in zip(bars, rates):
            ax.annotate(f"{r:.1%}", xy=(bar.get_x() + bar.get_width() / 2, r),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(results_dir, "plots", "model_vs_model.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Model vs model: {out_path}")


def generate_all_plots(results_dir: str = "results"):
    """Generuje wszystkie wykresy na podstawie zapisanych wyników."""
    plot_learning_curves(["ppo_a", "ppo_b", "trpo"], results_dir)

    eval_path = os.path.join(results_dir, "eval_results.json")
    if not os.path.exists(eval_path):
        print(f"Brak {eval_path} – pomiń wykresy win rate.")
        return

    with open(eval_path) as f:
        eval_results = json.load(f)

    plot_win_rates(eval_results, results_dir)
    plot_model_vs_model(eval_results, results_dir)
