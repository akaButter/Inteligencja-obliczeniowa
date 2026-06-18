"""Trening MaskablePPO i TRPO na środowisku Makao.

Strategia: jeden agent (player_0) uczy się przez RL; pozostali grają losowo.
Po treningu ten sam model używany jest przez wszystkich agentów w ewaluacji
(parametr sharing – wspólna polityka dla wszystkich graczy).
"""
import os
import csv
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib import MaskablePPO, TRPO
from sb3_contrib.common.wrappers import ActionMasker

from makao_env import MakaoEnv, ACTION_DRAW

# ---------------------------------------------------------------------------
# Konfiguracje hiperparametrów
# ---------------------------------------------------------------------------

def _lr_schedule(initial_lr: float):
    """Liniowy harmonogram LR: spada od initial_lr do initial_lr/10."""
    def func(progress_remaining: float) -> float:
        return max(initial_lr * progress_remaining, initial_lr / 10)
    return func


# Mapowanie: nazwa → (klasa algorytmu, użyj_maski, hiperparametry)
# use_mask=True: środowisko owinięte ActionMasker (tylko dla MaskablePPO)
# use_mask=False: TRPO uczy się bez masek; nieprawidłowe akcje są korygowane w env
ALGO_CONFIGS: dict[str, tuple] = {
    # PPO stabilny: duże rollouts, liniowy LR, wysoka gamma
    "ppo_a": (MaskablePPO, True, dict(
        learning_rate=_lr_schedule(3e-4),
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        ent_coef=0.05,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]},
    )),
    # PPO agresywny: krótkie rollouts, silna entropia
    "ppo_b": (MaskablePPO, True, dict(
        learning_rate=_lr_schedule(5e-4),
        n_steps=2048,
        batch_size=256,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.92,
        ent_coef=0.1,
        clip_range=0.25,
        policy_kwargs={"net_arch": [128, 128]},
    )),
    # TRPO: naturalna gradientowa optymalizacja z ograniczeniem KL (bez clippingu)
    # Nie wspiera natywnie masek akcji – nieprawidłowe akcje korygowane w env
    "trpo": (TRPO, False, dict(
        learning_rate=_lr_schedule(1e-3),   # tylko dla krytyka (wartości)
        n_steps=4096,                        # duży rollout → lepsza estymata nat. gradientu
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,                      # rozmiar trust region
        n_critic_updates=20,                 # więcej aktualizacji krytyka per rollout
        cg_max_steps=15,                     # iteracje conjugate gradient
        cg_damping=0.1,
        line_search_shrinking_factor=0.8,
        policy_kwargs={"net_arch": [128, 128]},
    )),
}


# ---------------------------------------------------------------------------
# Wrapper single-agent wokół AEC
# ---------------------------------------------------------------------------

class MakaoSingleAgentEnv(gym.Env):
    """Środowisko gym dla jednego agenta w Makao.

    Player_0 jest trenowanym agentem; pozostali gracze grają losowo.
    Umożliwia trening z MaskablePPO via ActionMasker.
    """

    metadata = {"render_modes": []}

    def __init__(self, agent_id: str = "player_0"):
        super().__init__()
        self._agent_id = agent_id
        self._aec = MakaoEnv()
        self.observation_space: Box = self._aec.observation_spaces[agent_id]
        self.action_space: Discrete = self._aec.action_spaces[agent_id]
        self._current_mask = np.ones(self.action_space.n, dtype=np.int8)
        self._current_mask[ACTION_DRAW] = 1

    def reset(self, seed=None, options=None):
        self._aec.reset(seed=seed)
        self._advance_to_our_turn()
        obs = self._aec.observe(self._agent_id)
        self._current_mask = self._aec.action_mask()
        return obs, {}

    def step(self, action):
        step_reward = 0.0
        terminated = False
        truncated = False

        # Koryguj nieprawidłowe akcje (dla TRPO który nie używa masek)
        action = int(action)
        if not self._current_mask[action]:
            valid = np.where(self._current_mask)[0]
            action = int(np.random.choice(valid)) if len(valid) > 0 else ACTION_DRAW

        # Wykonaj akcję naszego agenta
        self._aec.step(action)
        step_reward += self._aec.rewards.get(self._agent_id, 0.0)
        terminated = self._aec.terminations.get(self._agent_id, False)
        truncated = self._aec.truncations.get(self._agent_id, False)

        if terminated or truncated or not self._aec.agents:
            obs = self._aec.observe(self._agent_id)
            self._current_mask = self._fallback_mask()
            return obs, step_reward, True, False, {}

        # Pozwól pozostałym agentom zagrać (zachłannie losowo) aż do naszej kolejki
        while self._aec.agents and self._aec.agent_selection != self._agent_id:
            mask = self._aec.action_mask()
            play_valid = np.where(mask[:52])[0]
            if len(play_valid) > 0:
                rand_action = int(np.random.choice(play_valid))
            else:
                rand_action = int(np.random.choice(np.where(mask)[0]))
            self._aec.step(rand_action)
            step_reward += self._aec.rewards.get(self._agent_id, 0.0)
            terminated = self._aec.terminations.get(self._agent_id, False)
            truncated = self._aec.truncations.get(self._agent_id, False)
            if terminated or truncated or not self._aec.agents:
                obs = self._aec.observe(self._agent_id)
                self._current_mask = self._fallback_mask()
                return obs, step_reward, True, False, {}

        obs = self._aec.observe(self._agent_id)
        self._current_mask = self._aec.action_mask()
        return obs, step_reward, False, False, {}

    def get_action_mask(self) -> np.ndarray:
        """Zwraca maskę ważnych akcji (wymagana przez ActionMasker)."""
        return self._current_mask.astype(bool)

    def _advance_to_our_turn(self):
        """Pomija zachłannie losowe akcje innych agentów aż do pierwszej kolejki naszego agenta."""
        while self._aec.agents and self._aec.agent_selection != self._agent_id:
            mask = self._aec.action_mask()
            play_valid = np.where(mask[:52])[0]
            if len(play_valid) > 0:
                action = int(np.random.choice(play_valid))
            else:
                action = int(np.random.choice(np.where(mask)[0]))
            self._aec.step(action)

    def _fallback_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[ACTION_DRAW] = 1
        return mask

    def close(self):
        self._aec.close()


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class EpisodeRewardCallback(BaseCallback):
    """Zapisuje nagrody epizodów do CSV."""

    def __init__(self, csv_path: str):
        super().__init__()
        self._csv_path = csv_path
        self._rows: list[tuple] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self._rows.append((self.num_timesteps, ep["r"], ep["l"]))
        return True

    def _on_training_end(self):
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "ep_reward", "ep_length"])
            writer.writerows(self._rows)


# ---------------------------------------------------------------------------
# Tworzenie VecEnv
# ---------------------------------------------------------------------------

def _make_env_masked():
    env = MakaoSingleAgentEnv()
    env = ActionMasker(env, lambda e: e.get_action_mask())
    return env


def _make_env_unmasked():
    return MakaoSingleAgentEnv()


def make_vec_env(n_envs: int = 4, seed: int = 42, use_mask: bool = True):
    """Tworzy VecEnv (n_envs równoległych środowisk). use_mask=True dla MaskablePPO."""
    fn = _make_env_masked if use_mask else _make_env_unmasked
    vec_env = sb3_make_vec_env(fn, n_envs=n_envs, seed=seed)
    vec_env = VecMonitor(vec_env)
    return vec_env


# ---------------------------------------------------------------------------
# Trening
# ---------------------------------------------------------------------------

def train(
    config_name: str,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    seed: int = 42,
    results_dir: str = "results",
) -> str:
    """Trenuje MaskablePPO. Zwraca ścieżkę do zapisanego modelu (.zip)."""
    out_dir = os.path.join(results_dir, config_name)
    os.makedirs(out_dir, exist_ok=True)

    AlgoClass, use_mask, params = ALGO_CONFIGS[config_name]
    vec_env = make_vec_env(n_envs=n_envs, seed=seed, use_mask=use_mask)

    print(f"Trening: {config_name.upper()} | {total_timesteps:,} kroków | {n_envs} envs")
    model = AlgoClass(
        "MlpPolicy",
        vec_env,
        tensorboard_log=os.path.join(out_dir, "tb"),
        verbose=0,
        seed=seed,
        **params,
    )

    csv_path = os.path.join(out_dir, "learning_curve.csv")
    model.learn(
        total_timesteps=total_timesteps,
        callback=EpisodeRewardCallback(csv_path),
        progress_bar=True,
    )

    model_path = os.path.join(out_dir, "model")
    model.save(model_path)
    vec_env.close()
    print(f"[{config_name}] Zapisano: {model_path}.zip")
    return model_path + ".zip"
