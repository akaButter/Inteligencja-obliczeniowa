from __future__ import annotations

from typing import Any, Dict, Type

from stable_baselines3 import PPO, SAC, TD3


ALGO_CLASSES: Dict[str, Type] = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def build_model(
    algo_name: str,
    env,
    params: Dict[str, Any],
    policy_kwargs: Dict[str, Any] | None,
    seed: int,
    device: str,
):
    """Create a Stable-Baselines3 model for the requested algorithm."""

    if algo_name not in ALGO_CLASSES:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    algo_cls = ALGO_CLASSES[algo_name]
    kwargs = dict(params)
    if algo_name == "TD3" and "use_sde" in kwargs:
        kwargs.pop("use_sde")
    if policy_kwargs:
        policy_kwargs_clean = dict(policy_kwargs)
        if algo_name == "TD3" and "use_sde" in policy_kwargs_clean:
            policy_kwargs_clean.pop("use_sde")
        kwargs["policy_kwargs"] = policy_kwargs_clean

    return algo_cls(
        "MlpPolicy",
        env,
        seed=seed,
        device=device,
        **kwargs,
    )