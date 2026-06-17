"""Ewaluacja wytrenowanych modeli: scenariusz jednorodny i mieszany."""
import json
import os
import numpy as np
from pettingzoo.utils import wrappers
from sb3_contrib import MaskablePPO

from makao_env import MakaoEnv


def _make_eval_env():
    env = MakaoEnv()
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def _random_action(mask: np.ndarray) -> int:
    """Zachłanny losowy agent: zagraj kartę jeśli możliwe, dobieraj tylko gdy musisz."""
    play_valid = np.where(mask[:52])[0]
    if len(play_valid) > 0:
        return int(np.random.choice(play_valid))
    valid = np.where(mask)[0]
    return int(np.random.choice(valid))


def _find_winner(env) -> str | None:
    """Zwraca identyfikator zwycięzcy na podstawie ostatnich nagród."""
    raw = env.unwrapped
    for a in raw.possible_agents:
        if raw.terminations.get(a, False) and raw.rewards.get(a, 0.0) > 0.0:
            return a
    return None


def evaluate_all_ppo(model_path: str, n_episodes: int = 200) -> dict:
    """Scenariusz jednorodny: wszyscy 4 agenci używają tego samego modelu PPO."""
    model = MaskablePPO.load(model_path)
    env = _make_eval_env()
    wins = {a: 0 for a in env.unwrapped.possible_agents}
    timeouts = 0

    for _ in range(n_episodes):
        env.reset()
        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue
            mask = env.unwrapped.action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            env.step(int(action))

        winner = _find_winner(env)
        if winner:
            wins[winner] += 1
        else:
            timeouts += 1

    env.close()
    win_rates = {a: wins[a] / n_episodes for a in wins}
    return {"wins": wins, "win_rates": win_rates, "timeouts": timeouts}


def evaluate_mixed(
    model_path: str,
    ppo_agents: list[str] | None = None,
    n_episodes: int = 200,
) -> dict:
    """Scenariusz mieszany: część agentów gra PPO, reszta losowo.

    Domyślnie player_0 i player_1 używają PPO, player_2 i player_3 grają losowo.
    """
    if ppo_agents is None:
        ppo_agents = ["player_0", "player_1"]

    model = MaskablePPO.load(model_path)
    env = _make_eval_env()
    wins = {a: 0 for a in env.unwrapped.possible_agents}
    timeouts = 0

    for _ in range(n_episodes):
        env.reset()
        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue
            mask = env.unwrapped.action_mask()
            if agent in ppo_agents:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            else:
                action = _random_action(mask)
            env.step(action)

        winner = _find_winner(env)
        if winner:
            wins[winner] += 1
        else:
            timeouts += 1

    env.close()
    win_rates = {a: wins[a] / n_episodes for a in wins}
    ppo_total = sum(wins[a] for a in ppo_agents)
    random_agents = [a for a in wins if a not in ppo_agents]
    random_total = sum(wins[a] for a in random_agents)
    return {
        "wins": wins,
        "win_rates": win_rates,
        "ppo_win_rate": ppo_total / n_episodes,
        "random_win_rate": random_total / n_episodes,
        "timeouts": timeouts,
    }


def evaluate_random_baseline(n_episodes: int = 200) -> dict:
    """Baseline: wszyscy agenci grają losowo."""
    env = _make_eval_env()
    wins = {a: 0 for a in env.unwrapped.possible_agents}
    timeouts = 0

    for _ in range(n_episodes):
        env.reset()
        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue
            mask = env.unwrapped.action_mask()
            env.step(_random_action(mask))

        winner = _find_winner(env)
        if winner:
            wins[winner] += 1
        else:
            timeouts += 1

    env.close()
    win_rates = {a: wins[a] / n_episodes for a in wins}
    return {"wins": wins, "win_rates": win_rates, "timeouts": timeouts}


def run_all_evaluations(
    results_dir: str = "results",
    n_episodes: int = 200,
) -> dict:
    """Uruchamia pełną ewaluację dla ppo_a, ppo_b i baseline losowego."""
    all_results = {}

    print("Ewaluacja baseline losowego...")
    all_results["random_baseline"] = evaluate_random_baseline(n_episodes)

    for config in ("ppo_a", "ppo_b"):
        model_path = os.path.join(results_dir, config, "model.zip")
        if not os.path.exists(model_path):
            print(f"Brak modelu {model_path}, pomijam.")
            continue

        print(f"Ewaluacja {config} – scenariusz jednorodny...")
        all_results[f"{config}_all_ppo"] = evaluate_all_ppo(model_path, n_episodes)

        print(f"Ewaluacja {config} – scenariusz mieszany (2 PPO vs 2 losowych)...")
        all_results[f"{config}_mixed"] = evaluate_mixed(model_path, n_episodes=n_episodes)

    out_path = os.path.join(results_dir, "eval_results.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wyniki zapisane: {out_path}")
    return all_results
