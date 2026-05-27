from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt


def plot_learning_curves(curves: Dict[str, Dict], title: str, out_path: str) -> None:
    """Plot mean and standard deviation for multiple configurations."""

    plt.figure(figsize=(10, 6))

    for label, curve in curves.items():
        timesteps = curve["timesteps"]
        mean = curve["mean"]
        std = curve["std"]
        line = plt.plot(timesteps, mean, label=label, linewidth=2)
        color = line[0].get_color()
        plt.fill_between(timesteps, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color=color, alpha=0.15)

    plt.axhline(200, color="green", linestyle="--", alpha=0.7, label="Success threshold (200)")
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
