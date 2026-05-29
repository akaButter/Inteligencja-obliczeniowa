from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List, Tuple

from analysis import aggregate_curves, compute_summary, save_json
from callbacks import EpisodeStatsCallback
from env_utils import make_env
from models import build_model


def train_single_run(
    algo_name: str,
    env_id: str,
    params: Dict[str, Any],
    policy_kwargs: Dict[str, Any] | None,
    seed: int,
    total_timesteps: int,
    device: str,
    run_dir: str,
) -> Dict[str, Any]:
    """Train one run and return raw episode history and timing."""

    print(f"    [run] seed={seed} timesteps={total_timesteps}")
    os.makedirs(run_dir, exist_ok=True)
    monitor_path = os.path.join(run_dir, "monitor.csv")

    env = make_env(env_id, seed=seed, monitor_path=monitor_path)
    model = build_model(
        algo_name=algo_name,
        env=env,
        params=params,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
    )

    callback = EpisodeStatsCallback()
    start_time = time.perf_counter()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    except ImportError:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    end_time = time.perf_counter()

    duration = end_time - start_time
    steps_done = max(1, int(model.num_timesteps))
    episodes_done = max(1, len(callback.episode_rewards))

    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "length", "global_step"])
        for idx, (reward, length, timestep) in enumerate(
            zip(
                callback.episode_rewards,
                callback.episode_lengths,
                callback.episode_timesteps,
            )
        ):
            writer.writerow([idx, reward, length, timestep])

    env.close()

    # if algo_name == "PPO":
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)

    print(
        "    [done] duration={:.2f}s, time/step={:.6f}s, time/episode={:.4f}s".format(
            duration, duration / steps_done, duration / episodes_done
        )
    )

    return {
        "timesteps": callback.episode_timesteps,
        "rewards": callback.episode_rewards,
        "lengths": callback.episode_lengths,
        "duration_sec": duration,
        "time_per_step_sec": duration / steps_done,
        "time_per_episode_sec": duration / episodes_done,
        "mean_reward": float(sum(callback.episode_rewards) / episodes_done),
    }


def train_config(
    algo_name: str,
    env_id: str,
    params: Dict[str, Any],
    policy_kwargs: Dict[str, Any] | None,
    total_timesteps: int,
    n_runs: int,
    device: str,
    seed_offset: int,
    curve_points: int,
    output_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Train multiple runs for a single hyperparam set and aggregate results."""

    os.makedirs(output_dir, exist_ok=True)

    print(f"[config] {algo_name} -> {os.path.basename(output_dir)}")

    run_histories: List[Dict[str, Any]] = []
    all_rewards: List[float] = []
    all_lengths: List[int] = []
    run_timings: List[Dict[str, float]] = []

    for run_idx in range(n_runs):
        seed = seed_offset + run_idx
        run_dir = os.path.join(output_dir, "runs", f"run_{run_idx}")
        history = train_single_run(
            algo_name=algo_name,
            env_id=env_id,
            params=params,
            policy_kwargs=policy_kwargs,
            seed=seed,
            total_timesteps=total_timesteps,
            device=device,
            run_dir=run_dir,
        )

        run_histories.append(
            {
                "timesteps": history["timesteps"],
                "rewards": history["rewards"],
            }
        )
        all_rewards.extend(history["rewards"])
        all_lengths.extend(history["lengths"])
        run_timings.append(
            {
                "duration_sec": history["duration_sec"],
                "time_per_step_sec": history["time_per_step_sec"],
                "time_per_episode_sec": history["time_per_episode_sec"],
            }
        )

    curve = aggregate_curves(
        run_histories=run_histories,
        total_timesteps=total_timesteps,
        curve_points=curve_points,
    )

    summary = compute_summary(all_rewards, all_lengths)
    summary.update(
        {
            "algorithm": algo_name,
            "total_timesteps_per_run": total_timesteps,
            "n_runs": n_runs,
            "avg_total_duration_sec": float(
                sum(t["duration_sec"] for t in run_timings) / n_runs
            ),
            "avg_time_per_timestep_sec": float(
                sum(t["time_per_step_sec"] for t in run_timings) / n_runs
            ),
            "avg_time_per_episode_sec": float(
                sum(t["time_per_episode_sec"] for t in run_timings) / n_runs
            ),
        }
    )

    save_json(os.path.join(output_dir, "summary.json"), summary)
    save_json(os.path.join(output_dir, "curve.json"), curve)

    print("[config] saved summary.json and curve.json")

    return summary, curve, run_timings
