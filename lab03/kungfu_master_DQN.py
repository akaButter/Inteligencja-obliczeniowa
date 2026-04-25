import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    RecordEpisodeStatistics, GrayscaleObservation, ResizeObservation,
    FrameStackObservation, RecordVideo,
)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import json
import csv
from collections import deque
from tqdm import tqdm


class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = x / 255.0  # Normalizacja pikseli z [0,255] do [0,1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


SCRIPT_DIR = os.path.dirname(__file__) or "."
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "kungfu_dqn")
DEFAULT_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos", "kungfu_dqn")


def train_dqn(num_episodes=300, gamma=0.9, lr=2.5e-4, batch_size=32,
              memory_size=10000, target_update=1000,
              save_dir=DEFAULT_RESULTS_DIR, video_dir=DEFAULT_VIDEO_DIR):
    EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.1, 0.995

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    gym.register_envs(ale_py)
    env = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array")

    record_interval = max(1, num_episodes // 4)
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(
        env, video_folder=video_dir, name_prefix="dqn-training",
        episode_trigger=lambda ep: ep % record_interval == 0,
        disable_logger=True,
    )
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)
    env = RecordEpisodeStatistics(env)

    action_size = env.action_space.n

    online_net = DQN(action_size).to(device)
    target_net = DQN(action_size).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    memory = deque(maxlen=memory_size)
    epsilon = EPSILON_START
    rewards_history, lengths_history, loss_history = [], [], []
    global_step = 0

    print(f"[DQN] device={device} | episodes={num_episodes} | gamma={gamma}")
    print(f"[DQN] target update co {target_update} kroków, batch={batch_size}")

    pbar = tqdm(range(num_episodes), desc="DQN", unit="ep")
    for ep in pbar:
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32)
        done = False
        ep_loss = []

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                    action = int(online_net(obs_t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            done = terminated or truncated

            # Reward clipping do [-1, 0, +1]
            clipped = float(np.sign(reward))
            memory.append((obs, action, clipped, next_obs, float(done)))
            obs = next_obs
            global_step += 1

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                b_obs = torch.from_numpy(np.array([b[0] for b in batch], dtype=np.float32)).to(device)
                b_act = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
                b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                b_next = torch.from_numpy(np.array([b[3] for b in batch], dtype=np.float32)).to(device)
                b_done = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

                curr_q = online_net(b_obs).gather(1, b_act).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(b_next).max(dim=1)[0]
                    target_q = b_rew + gamma * max_next_q * (1.0 - b_done)

                # Huber loss zamiast MSE.
                loss = F.smooth_l1_loss(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                # POPRAWKA #4: gradient clipping.
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()
                ep_loss.append(loss.item())

                if global_step % target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if "episode" in info:
            ed = info["episode"]
            r = ed["r"][0] if isinstance(ed["r"], (list, np.ndarray)) else ed["r"]
            l = ed["l"][0] if isinstance(ed["l"], (list, np.ndarray)) else ed["l"]
            rewards_history.append(float(r))
            lengths_history.append(int(l))
            loss_history.append(float(np.mean(ep_loss)) if ep_loss else 0.0)

            if (ep + 1) % 5 == 0:
                avg_rew = np.mean(rewards_history[-25:])
                pbar.set_postfix({
                    "śr_25": f"{avg_rew:.0f}",
                    "max": f"{max(rewards_history):.0f}",
                    "eps": f"{epsilon:.3f}",
                })

    env.close()

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "length", "loss"])
        for i, (r, l, ls) in enumerate(zip(rewards_history, lengths_history, loss_history)):
            w.writerow([i, r, l, ls])

    summary = compute_summary(rewards_history, lengths_history, loss_history, gamma)
    summary["algorithm"] = "DQN"
    summary["num_episodes"] = num_episodes
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_dqn_curves(rewards_history, lengths_history, loss_history,
                    title=f"DQN (Kung-Fu Master) γ={gamma}",
                    out_path=os.path.join(save_dir, "curve.png"))
    print_summary(summary)

    torch.save(online_net.state_dict(), os.path.join(save_dir, "model.pt"))
    return rewards_history, lengths_history, summary


def compute_summary(rewards, lengths, losses, gamma):
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    n = len(rewards)
    last_quarter = rewards[3 * n // 4:] if n >= 4 else rewards
    first_quarter = rewards[: max(1, n // 4)]
    return {
        "gamma": gamma,
        "best_reward": float(np.max(rewards)) if n else 0.0,
        "worst_reward": float(np.min(rewards)) if n else 0.0,
        "mean_reward": float(np.mean(rewards)) if n else 0.0,
        "std_reward": float(np.std(rewards)) if n else 0.0,
        "mean_first_25%": float(np.mean(first_quarter)) if n else 0.0,
        "mean_last_25%": float(np.mean(last_quarter)) if n else 0.0,
        "improvement": float(np.mean(last_quarter) - np.mean(first_quarter)) if n else 0.0,
        "mean_length": float(np.mean(lengths)) if n else 0.0,
        "mean_loss": float(np.mean([l for l in losses if l > 0])) if any(l > 0 for l in losses) else 0.0,
        "discounted_return_first_1000_steps": float(
            sum(r * (gamma ** i) for i, r in enumerate(rewards[:1000]))
        ),
    }


def print_summary(s):
    print("\n" + "=" * 60)
    print(f" PODSUMOWANIE: {s.get('algorithm','')} (γ={s.get('gamma')})")
    print("=" * 60)
    print(f"  Najlepszy epizod:               {s['best_reward']:.0f} pkt")
    print(f"  Najgorszy epizod:               {s['worst_reward']:.0f} pkt")
    print(f"  Średnia (cały trening):         {s['mean_reward']:.1f} ± {s['std_reward']:.1f}")
    print(f"  Średnia (pierwsze 25%):         {s['mean_first_25%']:.1f}")
    print(f"  Średnia (ostatnie 25%):         {s['mean_last_25%']:.1f}")
    print(f"  Poprawa (last - first):         {s['improvement']:+.1f}")
    print(f"  Średnia długość epizodu:        {s['mean_length']:.0f} kroków")
    print(f"  Średnia strata (Huber):         {s['mean_loss']:.4f}")
    print(f"  Zdyskontowana suma (1000 ep):   {s['discounted_return_first_1000_steps']:.1f}")
    if s["improvement"] > 200:
        verdict = "Agent UCZY SIĘ silnie - widać wyraźną progresję."
    elif s["improvement"] > 0:
        verdict = "Agent uczy się powoli, dłuższy trening prawdopodobnie pomoże."
    else:
        verdict = "Brak progresji - rozważ zmianę hiperparametrów lub większą liczbę epizodów."
    print(f"\n  Wniosek: {verdict}")
    print("=" * 60 + "\n")


def plot_dqn_curves(rewards, lengths, losses, title, out_path):
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    episodes = np.arange(len(rewards))
    window = max(5, len(rewards) // 20)

    axes[0].plot(episodes, rewards, alpha=0.3, color="#1f77b4", label="surowa")
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window - 1, len(rewards)), ma,
                     color="red", linewidth=2, label=f"śr. krocząca ({window})")
    axes[0].set_xlabel("Epizod"); axes[0].set_ylabel("Suma nagród")
    axes[0].set_title("Krzywa uczenia"); axes[0].grid(True, alpha=0.4); axes[0].legend()

    axes[1].plot(episodes, lengths, alpha=0.3, color="green", label="surowa")
    if len(lengths) >= window:
        ma = np.convolve(lengths, np.ones(window) / window, mode="valid")
        axes[1].plot(np.arange(window - 1, len(lengths)), ma,
                     color="darkgreen", linewidth=2, label=f"śr. krocząca ({window})")
    axes[1].set_xlabel("Epizod"); axes[1].set_ylabel("Liczba kroków")
    axes[1].set_title("Długość epizodów"); axes[1].grid(True, alpha=0.4); axes[1].legend()

    axes[2].plot(episodes, losses, alpha=0.5, color="purple", label="strata Huber")
    axes[2].set_xlabel("Epizod"); axes[2].set_ylabel("Loss")
    axes[2].set_title("Strata sieci (lower = better)")
    axes[2].grid(True, alpha=0.4); axes[2].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    train_dqn()
