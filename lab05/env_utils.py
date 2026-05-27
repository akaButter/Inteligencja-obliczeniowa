from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


def make_env(env_id: str, seed: int, monitor_path: str | None = None):
    """Create a Gymnasium environment with optional Monitor wrapper."""

    env = gym.make(env_id)
    env.reset(seed=seed)
    if monitor_path:
        env = Monitor(env, monitor_path)
    return env
