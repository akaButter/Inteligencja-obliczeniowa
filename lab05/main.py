from __future__ import annotations

import os

os.environ.setdefault("OBJC_DISABLE_CLASS_DUPLICATE_CHECK", "YES")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import sys
import warnings
from typing import Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


# from analysis import save_json
from callbacks import EpisodeStatsCallback
from config import ExperimentConfig, default_architectures, default_hyperparams
from env_utils import make_env
from evaluate import evaluate_model
from models import build_model
from plots import plot_learning_curves
from train import train_config
import json
from pathlib import Path
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def save_json(path, data):
    # Konwertujemy ścieżkę na obiekt Path i wyciągamy z niej sam folder nadrzędny (.parent)
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Teraz bezpiecznie otwieramy plik
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def _write_params(output_dir: str, algo_name: str, params: Dict, policy_kwargs: Dict | None) -> None:
    payload = {
        "algorithm": algo_name,
        "params": params,
        "policy_kwargs": policy_kwargs or {},
    }
    save_json(os.path.join(output_dir, "params.json"), payload)


def _select_best_config(summaries: Dict[str, Dict]) -> Tuple[str, Dict]:
    best_name = None
    best_summary = None
    for name, summary in summaries.items():
        score = summary.get("mean_last_25%", float("-inf"))
        if best_summary is None or score > best_summary.get("mean_last_25%", float("-inf")):
            best_name = name
            best_summary = summary
    return best_name, best_summary


def run_hyperparam_sweeps(cfg: ExperimentConfig) -> Dict[str, Dict[str, Dict]]:
    hyperparams = default_hyperparams()
    architectures = default_architectures()
    base_arch = architectures[0]

    all_summaries: Dict[str, Dict[str, Dict]] = {}

    for algo_name in cfg.algorithms:
        algo_dir = os.path.join(cfg.results_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        curves: Dict[str, Dict] = {}
        summaries: Dict[str, Dict] = {}

        print(f"[algo] {algo_name} hyperparam sweep")

        for hp in hyperparams[algo_name]:
            output_dir = os.path.join(algo_dir, hp.name)
            _write_params(output_dir, algo_name, hp.params, base_arch.policy_kwargs)

            summary, curve, _ = train_config(
                algo_name=algo_name,
                env_id=cfg.env_id,
                params=hp.params,
                policy_kwargs=base_arch.policy_kwargs,
                total_timesteps=cfg.total_timesteps,
                n_runs=cfg.n_runs,
                device=cfg.device,
                seed_offset=cfg.seed_offset,
                curve_points=cfg.curve_points,
                output_dir=output_dir,
            )

            curves[hp.name] = curve
            summaries[hp.name] = summary

        chart_path = os.path.join(algo_dir, "comparison_curve.png")
        plot_learning_curves(
            curves,
            title=f"{algo_name} on {cfg.env_id} (mean ± std, {cfg.n_runs} runs)",
            out_path=chart_path,
        )
        print(f"[algo] saved curve: {chart_path}")

        all_summaries[algo_name] = summaries

    return all_summaries


def run_architecture_comparison(cfg: ExperimentConfig, algo_name: str) -> None:
    hyperparams = default_hyperparams()[algo_name][0]
    architectures = default_architectures()
    compare_dir = os.path.join(cfg.results_dir, "arch_compare", algo_name)
    os.makedirs(compare_dir, exist_ok=True)

    curves: Dict[str, Dict] = {}

    print(f"[arch] {algo_name} architecture comparison")

    for arch in architectures:
        output_dir = os.path.join(compare_dir, arch.name)
        _write_params(output_dir, algo_name, hyperparams.params, arch.policy_kwargs)

        _, curve, _ = train_config(
            algo_name=algo_name,
            env_id=cfg.env_id,
            params=hyperparams.params,
            policy_kwargs=arch.policy_kwargs,
            total_timesteps=cfg.total_timesteps,
            n_runs=cfg.n_runs,
            device=cfg.device,
            seed_offset=cfg.seed_offset,
            curve_points=cfg.curve_points,
            output_dir=output_dir,
        )

        curves[arch.name] = curve

    chart_path = os.path.join(compare_dir, "comparison_curve.png")
    plot_learning_curves(
        curves,
        title=f"{algo_name} architecture comparison",
        out_path=chart_path,
    )
    print(f"[arch] saved curve: {chart_path}")


def train_and_save_best_model(
    cfg: ExperimentConfig,
    algo_name: str,
    hp_name: str,
    params: Dict,
    policy_kwargs: Dict | None,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    env = make_env(cfg.env_id, seed=cfg.seed_offset, monitor_path=None)
    model = build_model(
        algo_name=algo_name,
        env=env,
        params=params,
        policy_kwargs=policy_kwargs,
        seed=cfg.seed_offset,
        device=cfg.device,
    )
    callback = EpisodeStatsCallback()
    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=False)
    except ImportError:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

    model_path = os.path.join(output_dir, "best_model.zip")
    model.save(model_path)
    env.close()

    metadata = {
        "algorithm": algo_name,
        "hyperparam_set": hp_name,
        "total_timesteps": cfg.total_timesteps,
    }
    save_json(os.path.join(output_dir, "best_model_info.json"), metadata)
    return model_path


def run_best_model_eval(cfg: ExperimentConfig, summaries: Dict[str, Dict[str, Dict]]) -> None:
    hyperparams = default_hyperparams()
    architectures = default_architectures()
    base_arch = architectures[0]

    best_algo = None
    best_hp = None
    best_summary = None

    for algo_name, algo_summaries in summaries.items():
        hp_name, summary = _select_best_config(algo_summaries)
        if summary is None:
            continue
        if best_summary is None or summary.get("mean_last_25%", float("-inf")) > best_summary.get(
            "mean_last_25%", float("-inf")
        ):
            best_algo = algo_name
            best_hp = hp_name
            best_summary = summary

    if best_algo is None or best_hp is None:
        return

    print(f"[best] selected {best_algo}:{best_hp}")

    params = next(hp.params for hp in hyperparams[best_algo] if hp.name == best_hp)
    best_dir = os.path.join(cfg.results_dir, "best_model")
    model_path = train_and_save_best_model(
        cfg=cfg,
        algo_name=best_algo,
        hp_name=best_hp,
        params=params,
        policy_kwargs=base_arch.policy_kwargs,
        output_dir=best_dir,
    )

    eval_results = evaluate_model(
        algo_name=best_algo,
        env_id=cfg.env_id,
        model_path=model_path,
        n_eval_episodes=10,
        seed=cfg.seed_offset,
    )
    save_json(os.path.join(best_dir, "eval.json"), eval_results)
    print("[best] saved eval.json")



def run_all(cfg: ExperimentConfig) -> None:
    manifest = {
        "env_id": cfg.env_id,
        "total_timesteps": cfg.total_timesteps,
        "n_runs": cfg.n_runs,
        "algorithms": cfg.algorithms,
    }
    save_json(os.path.join(cfg.results_dir, "manifest.json"), manifest)
    print(f"[run] results dir: {cfg.results_dir}")

    summaries = run_hyperparam_sweeps(cfg)
    run_architecture_comparison(cfg, algo_name="SAC")
    run_best_model_eval(cfg, summaries)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="LunarLander continuous control experiments")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--env-id", default="LunarLanderContinuous-v3")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--curve-points", type=int, default=500)
    parser.add_argument("--algorithms", nargs="+", default=["PPO", "SAC", "TD3"])

    args = parser.parse_args()
    return ExperimentConfig(
        results_dir=args.results_dir,
        env_id=args.env_id,
        total_timesteps=args.timesteps,
        n_runs=args.runs,
        device=args.device,
        seed_offset=args.seed_offset,
        curve_points=args.curve_points,
        algorithms=args.algorithms,
    )


def main() -> None:
    cfg = parse_args()
    run_all(cfg)


if __name__ == "__main__":
    main()
