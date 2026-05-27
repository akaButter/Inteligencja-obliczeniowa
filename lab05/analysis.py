from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np


def compute_summary(rewards: List[float], lengths: List[int]) -> Dict[str, Any]:
    """Compute aggregate statistics over all collected episodes."""

    rewards_arr = np.array(rewards, dtype=float)
    lengths_arr = np.array(lengths, dtype=float)
    n = len(rewards_arr)
    if n == 0:
        return {"total_tracked_episodes": 0}

    first_q = rewards_arr[: max(1, n // 4)]
    last_q = rewards_arr[3 * n // 4 :]

    return {
        "total_tracked_episodes": int(n),
        "best_reward": float(np.max(rewards_arr)),
        "worst_reward": float(np.min(rewards_arr)),
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_first_25%": float(np.mean(first_q)),
        "mean_last_25%": float(np.mean(last_q)),
        "improvement": float(np.mean(last_q) - np.mean(first_q)),
        "mean_length": float(np.mean(lengths_arr)),
        "solved_episodes": int(np.sum(rewards_arr > 200)),
        "solved_pct": float(100 * np.sum(rewards_arr > 200) / n),
    }


def aggregate_curves(
    run_histories: List[Dict[str, List[float]]],
    total_timesteps: int,
    curve_points: int,
) -> Dict[str, List[float]]:
    """Interpolate episode rewards over a shared timestep grid."""

    common_timesteps = np.linspace(0, total_timesteps, curve_points)
    interpolated = []

    for history in run_histories:
        t_seq = history["timesteps"]
        r_seq = history["rewards"]
        full_t = [0.0] + list(t_seq)
        full_r = [0.0] + list(r_seq)
        interp_r = np.interp(common_timesteps, full_t, full_r)
        interpolated.append(interp_r)

    interpolated = np.array(interpolated)
    mean_curve = np.mean(interpolated, axis=0)
    std_curve = np.std(interpolated, axis=0)

    return {
        "timesteps": common_timesteps.tolist(),
        "mean": mean_curve.tolist(),
        "std": std_curve.tolist(),
    }


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """Serialize results to JSON."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
