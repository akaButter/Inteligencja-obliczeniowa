"""Ewaluacja wytrenowanych modeli: jednorodny, vs losowy, model vs model."""
import json
import os
import numpy as np
from pettingzoo.utils import wrappers
from sb3_contrib import MaskablePPO, TRPO

from makao_env import MakaoEnv, ACTION_DRAW

POSSIBLE_AGENTS = [f"player_{i}" for i in range(4)]


def _make_eval_env():
    env = MakaoEnv()
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def _random_action(mask: np.ndarray) -> int:
    """Zachłanny losowy: zagraj kartę jeśli możliwe, dobierz gdy musisz."""
    play_valid = np.where(mask[:52])[0]
    if len(play_valid) > 0:
        return int(np.random.choice(play_valid))
    return int(np.random.choice(np.where(mask)[0]))


def _find_winner(env) -> str | None:
    raw = env.unwrapped
    for a in raw.possible_agents:
        if raw.terminations.get(a, False) and raw.rewards.get(a, 0.0) > 0.0:
            return a
    return None


def _load_model(config_name: str, model_path: str):
    """Ładuje model właściwej klasy; zwraca (model, is_maskable)."""
    if config_name == "trpo":
        return TRPO.load(model_path), False
    return MaskablePPO.load(model_path), True


def _predict(model, obs: np.ndarray, mask: np.ndarray, is_maskable: bool) -> int:
    """Przewiduje akcję; dla TRPO koryguje ewentualnie nieprawidłowy wybór."""
    if is_maskable:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    else:
        action, _ = model.predict(obs, deterministic=True)
        if not mask[int(action)]:
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid)) if len(valid) > 0 else ACTION_DRAW
    return int(action)


def _run_episodes(agent_models: dict, n_episodes: int) -> dict:
    """Generyczna pętla ewaluacji.

    agent_models: {agent_id: (model, is_maskable) | None}
    None oznacza zachłannego losowego gracza.
    """
    env = _make_eval_env()
    wins = {a: 0 for a in POSSIBLE_AGENTS}
    timeouts = 0

    for _ in range(n_episodes):
        env.reset()
        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue
            mask = env.unwrapped.action_mask()
            entry = agent_models.get(agent)
            if entry is not None:
                action = _predict(entry[0], obs, mask, entry[1])
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
    return {"wins": wins, "win_rates": win_rates, "timeouts": timeouts}


# ---------------------------------------------------------------------------
# Scenariusze ewaluacji
# ---------------------------------------------------------------------------

def evaluate_random_baseline(n_episodes: int = 200) -> dict:
    """Baseline: wszyscy agenci grają losowo."""
    return _run_episodes({a: None for a in POSSIBLE_AGENTS}, n_episodes)


def evaluate_uniform(model_path: str, config_name: str, n_episodes: int = 200) -> dict:
    """Jednorodny: wszyscy 4 agenci używają tego samego modelu (parametr sharing)."""
    entry = _load_model(config_name, model_path)
    return _run_episodes({a: entry for a in POSSIBLE_AGENTS}, n_episodes)


def evaluate_vs_random(
    model_path: str, config_name: str, n_episodes: int = 200
) -> dict:
    """Model vs losowy: player_0+1 używają modelu, player_2+3 grają losowo."""
    entry = _load_model(config_name, model_path)
    result = _run_episodes(
        {"player_0": entry, "player_1": entry, "player_2": None, "player_3": None},
        n_episodes,
    )
    model_wins = result["wins"]["player_0"] + result["wins"]["player_1"]
    random_wins = result["wins"]["player_2"] + result["wins"]["player_3"]
    result["model_win_rate"] = model_wins / n_episodes
    result["random_win_rate"] = random_wins / n_episodes
    return result


def evaluate_model_vs_model(
    path_a: str, config_a: str,
    path_b: str, config_b: str,
    n_episodes: int = 200,
) -> dict:
    """Model vs Model: player_0+1 używają algo A, player_2+3 używają algo B."""
    entry_a = _load_model(config_a, path_a)
    entry_b = _load_model(config_b, path_b)
    result = _run_episodes(
        {"player_0": entry_a, "player_1": entry_a,
         "player_2": entry_b, "player_3": entry_b},
        n_episodes,
    )
    wins_a = result["wins"]["player_0"] + result["wins"]["player_1"]
    wins_b = result["wins"]["player_2"] + result["wins"]["player_3"]
    result["team_a_win_rate"] = wins_a / n_episodes
    result["team_b_win_rate"] = wins_b / n_episodes
    result["config_a"] = config_a
    result["config_b"] = config_b
    return result


def run_all_evaluations(
    results_dir: str = "results",
    n_episodes: int = 200,
) -> dict:
    """Pełna ewaluacja: baseline, jednorodny, vs losowy, model vs model."""
    all_results = {}
    configs = ("ppo_a", "ppo_b", "trpo")

    available = {}
    for config in configs:
        path = os.path.join(results_dir, config, "model.zip")
        if os.path.exists(path):
            available[config] = path
        else:
            print(f"Brak modelu {path}, pomijam.")

    print("Ewaluacja baseline losowego...")
    all_results["baseline_random"] = evaluate_random_baseline(n_episodes)

    for config, path in available.items():
        print(f"[{config.upper()}] Jednorodny (wszyscy {config.upper()})...")
        all_results[f"{config}_jednorodny"] = evaluate_uniform(path, config, n_episodes)

        print(f"[{config.upper()}] Model vs losowy...")
        all_results[f"{config}_vs_random"] = evaluate_vs_random(path, config, n_episodes)

    # Pairwise model vs model
    avail_list = list(available.items())
    for i, (c1, p1) in enumerate(avail_list):
        for c2, p2 in avail_list[i + 1:]:
            print(f"Model vs Model: {c1.upper()} vs {c2.upper()}...")
            all_results[f"{c1}_vs_{c2}"] = evaluate_model_vs_model(
                p1, c1, p2, c2, n_episodes
            )

    out_path = os.path.join(results_dir, "eval_results.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wyniki zapisane: {out_path}")
    return all_results
