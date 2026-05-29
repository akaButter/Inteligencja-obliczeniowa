from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


DEFAULT_ENV_ID = "LunarLanderContinuous-v2"


@dataclass(frozen=True)
class HyperparamSet:
    """Named set of hyperparameters for a single algorithm."""

    name: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class ArchitectureSpec:
    """Policy network architecture definition used by SB3."""

    name: str
    policy_kwargs: Dict[str, Any]
    description: str


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration shared across runs."""

    results_dir: str
    env_id: str = DEFAULT_ENV_ID
    total_timesteps: int = 50_000
    n_runs: int = 10
    device: str = "auto"
    seed_offset: int = 0
    curve_points: int = 500
    algorithms: List[str] = field(default_factory=lambda: ["PPO"])


def default_hyperparams() -> Dict[str, List[HyperparamSet]]:
    """Return minimal yet distinct hyperparam sets per algorithm."""

    return {
        "PPO": [
            HyperparamSet(
                name="ppo_a",
                params={
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "batch_size": 256,
                    "n_steps": 2048,
                    "ent_coef": 0.0,
                },
            ),
            HyperparamSet(
                name="ppo_b",
                params={
                    "learning_rate": 1e-3,
                    "gamma": 0.98,
                    "gae_lambda": 0.9,
                    "batch_size": 512,
                    "n_steps": 2048,
                    "ent_coef": 0.01,
                },
            ),
            HyperparamSet(
                name="ppo_c",
                params={
                    "learning_rate": 1e-4,
                    "gamma": 0.995,
                    "gae_lambda": 0.98,
                    "batch_size": 256,
                    "n_steps": 4096,
                    "ent_coef": 0.0,
                },
            ),
        ],
        "SAC": [
            HyperparamSet(
                name="sac_a",
                params={
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "batch_size": 256,
                    "buffer_size": 300_000,
                    "train_freq": 1,
                    "gradient_steps": 1,
                },
            ),
            HyperparamSet(
                name="sac_b",
                params={
                    "learning_rate": 1e-3,
                    "gamma": 0.99,
                    "tau": 0.01,
                    "batch_size": 512,
                    "buffer_size": 300_000,
                    "train_freq": 1,
                    "gradient_steps": 1,
                },
            ),
            HyperparamSet(
                name="sac_c",
                params={
                    "learning_rate": 1e-4,
                    "gamma": 0.995,
                    "tau": 0.005,
                    "batch_size": 256,
                    "buffer_size": 300_000,
                    "train_freq": 1,
                    "gradient_steps": 1,
                },
            ),
        ],
        "TD3": [
            HyperparamSet(
                name="td3_a",
                params={
                    "learning_rate": 1e-3,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "batch_size": 256,
                    "buffer_size": 300_000,
                    "train_freq": 1,
                    "gradient_steps": 1,
                },
            ),
            HyperparamSet(
                name="td3_b",
                params={
                    "learning_rate": 3e-4,
                    "gamma": 0.98,
                    "tau": 0.01,
                    "batch_size": 512,
                    "buffer_size": 300_000,
                    "train_freq": 1,
                    "gradient_steps": 1,
                },
            ),
            HyperparamSet(
                name="td3_c",
                params={
                    "learning_rate": 1e-4,
                    "gamma": 0.995,
                    "tau": 0.005,
                    "batch_size": 256,
                    "buffer_size": 300_000,
                    "train_freq": 1,
                    "gradient_steps": 1,
                },
            ),
        ],
    }


def default_architectures() -> List[ArchitectureSpec]:
    """Return two network architectures for the architecture comparison task."""

    return [
        ArchitectureSpec(
            name="small",
            policy_kwargs={"net_arch": [64, 64]},
            description="Two hidden layers, 64 units each, ReLU activations.",
        ),
        ArchitectureSpec(
            name="large",
            policy_kwargs={"net_arch": [256, 256]},
            description="Two hidden layers, 256 units each, ReLU activations.",
        ),
    ]
