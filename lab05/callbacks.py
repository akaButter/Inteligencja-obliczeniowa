from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    """Collect per-episode rewards, lengths, and global timesteps."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                self.episode_timesteps.append(int(self.num_timesteps))
        return True
