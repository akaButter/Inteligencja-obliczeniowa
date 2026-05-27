from __future__ import annotations

from typing import Dict, List

import numpy as np

from env_utils import make_env
from models import ALGO_CLASSES


def evaluate_model(
    algo_name: str,
    env_id: str,
    model_path: str,
    n_eval_episodes: int,
    seed: int,
) -> Dict[str, List[float]]:
    """Run deterministic evaluation episodes for a saved model."""

    if algo_name not in ALGO_CLASSES:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    model_cls = ALGO_CLASSES[algo_name]
    env = make_env(env_id, seed=seed, monitor_path=None)
    model = model_cls.load(model_path)

    rewards: List[float] = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated
        rewards.append(episode_reward)

    env.close()

    return {
        "rewards": rewards,
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
    }
