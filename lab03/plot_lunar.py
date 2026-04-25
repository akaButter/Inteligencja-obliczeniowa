import os
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(__file__) or "."
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def smooth(series, window=50):
    return series.rolling(window=window, min_periods=1).mean()


def plot_single_run(csv_path, title=None, window=50):
    df = pd.read_csv(csv_path)
    if "reward" not in df.columns:
        df = pd.read_csv(csv_path, skiprows=1)
        df = df.rename(columns={"r": "reward", "l": "length"})

    plt.figure(figsize=(10, 6))
    plt.plot(df["reward"], alpha=0.3, color="skyblue", label="surowa")
    plt.plot(smooth(df["reward"], window), color="darkblue", linewidth=2,
             label=f"śr. krocząca ({window})")
    plt.axhline(200, color="green", linestyle="--", alpha=0.6, label="próg sukcesu")
    plt.xlabel("Epizod"); plt.ylabel("Nagroda")
    plt.title(title or os.path.basename(csv_path))
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_compare(csv_paths_with_labels, title="Porównanie algorytmów (Lunar Lander)",
                 window=50, save_path=None):
    """csv_paths_with_labels: lista (path, label, color)."""
    plt.figure(figsize=(11, 6))
    for path, label, color in csv_paths_with_labels:
        if not os.path.exists(path):
            print(f"  pomijam (brak pliku): {path}")
            continue
        try:
            df = pd.read_csv(path)
            if "reward" not in df.columns:
                df = pd.read_csv(path, skiprows=1).rename(columns={"r": "reward"})
        except Exception as e:
            print(f"  błąd przy {path}: {e}")
            continue
        plt.plot(smooth(df["reward"], window), color=color, linewidth=2, label=label)
    plt.axhline(200, color="green", linestyle="--", alpha=0.5, label="próg sukcesu (200)")
    plt.axhline(0, color="black", linestyle="-", alpha=0.15)
    plt.xlabel("Epizod"); plt.ylabel(f"Nagroda (śr. krocząca, okno={window})")
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    runs = [
        (os.path.join(RESULTS_DIR, "lunar_ppo", "history.csv"), "PPO γ=0.9", "navy"),
        (os.path.join(RESULTS_DIR, "lunar_dqn", "history.csv"), "DQN γ=0.9", "darkred"),
    ]
    out = os.path.join(RESULTS_DIR, "lunar_experiments", "compare.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_compare(runs, save_path=out)
