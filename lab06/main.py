"""Punkt wejścia: trening, ewaluacja i generowanie wykresów dla gry Makao."""
import argparse
import sys
import os

# Zapewnia działanie importów niezależnie od CWD
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
os.chdir(_dir)

from train import train, ALGO_CONFIGS
from evaluate import run_all_evaluations
from plot import generate_all_plots


def train_all(
    timesteps: int = 1_000_000,
    n_envs: int = 4,
    seed: int = 42,
    results_dir: str = "results",
    configs: list[str] | None = None,
):
    """Trenuje wszystkie konfiguracje PPO (domyślnie ppo_a i ppo_b)."""
    if configs is None:
        configs = list(ALGO_CONFIGS.keys())
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Trening: {config} | {timesteps} kroków | {n_envs} środowisk")
        print(f"{'='*50}")
        train(
            config_name=config,
            total_timesteps=timesteps,
            n_envs=n_envs,
            seed=seed,
            results_dir=results_dir,
        )


def evaluate_all(results_dir: str = "results", n_episodes: int = 200):
    """Uruchamia ewaluację wszystkich wytrenowanych modeli."""
    print(f"\nEwaluacja: {n_episodes} epizodów na konfigurację")
    return run_all_evaluations(results_dir=results_dir, n_episodes=n_episodes)


def plot_results(results_dir: str = "results"):
    """Generuje wszystkie wykresy z zapisanych wyników."""
    print("\nGenerowanie wykresów")
    generate_all_plots(results_dir=results_dir)


def main():
    parser = argparse.ArgumentParser(description="Makao multi-agent RL")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "plot", "all"],
        default="all",
        help="Tryb: train | eval | plot | all",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Liczba kroków treningu")
    parser.add_argument("--n-envs", type=int, default=4, help="Liczba równoległych środowisk")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowości")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Liczba epizodów ewaluacji")
    parser.add_argument("--results-dir", default="results", help="Katalog z wynikami")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=list(ALGO_CONFIGS.keys()),
        help="Konfiguracje do treningu",
    )

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode in ("train", "all"):
        train_all(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            results_dir=args.results_dir,
            configs=args.configs,
        )

    if args.mode in ("eval", "all"):
        evaluate_all(results_dir=args.results_dir, n_episodes=args.eval_episodes)

    if args.mode in ("plot", "all"):
        plot_results(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
