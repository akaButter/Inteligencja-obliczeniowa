import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os
import json
import csv
import pickle
from tqdm import tqdm

# ==========================================
# 1. Konfiguracja Hiperparametrów
# ==========================================
NUM_EPISODES = 500
RECORD_INTERVAL = max(1, NUM_EPISODES // 4)

ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

SELECTED_RAM_BYTES = [36, 38, 40, 50, 80, 84, 100, 120]
BINS_PER_BYTE = 4


def discretize_state(ram_obs):
    arr = np.array(ram_obs).flatten()
    selected = arr[SELECTED_RAM_BYTES]
    binned = (selected // (256 // BINS_PER_BYTE)).astype(int)
    return tuple(binned.tolist())


SCRIPT_DIR = os.path.dirname(__file__) or "."
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "kungfu_qlearning")
DEFAULT_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos", "kungfu_qlearning")


def train_q_learning(num_episodes=NUM_EPISODES, gamma=GAMMA, alpha=ALPHA,
                     record_video=True, save_dir=DEFAULT_RESULTS_DIR,
                     video_dir=DEFAULT_VIDEO_DIR):
    gym.register_envs(ale_py)
    env = gym.make("ALE/KungFuMaster-v5", obs_type="ram", render_mode="rgb_array")

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env, video_folder=video_dir, name_prefix="qlearn-training",
            episode_trigger=lambda ep: ep % RECORD_INTERVAL == 0,
            disable_logger=True,
        )

    action_size = env.action_space.n
    q_table = defaultdict(lambda: np.zeros(action_size))

    epsilon = EPSILON_START
    rewards_history, lengths_history, epsilon_history = [], [], []
    best_episode_reward = -np.inf

    print(f"[Q-Learning] Trening: {num_episodes} epizodów, gamma={gamma}, alpha={alpha}")
    print(f"[Q-Learning] Stan = {len(SELECTED_RAM_BYTES)} bajtów RAM x {BINS_PER_BYTE} koszyków")

    pbar = tqdm(range(num_episodes), desc="Q-Learning", unit="ep")
    for episode in pbar:
        obs, _ = env.reset()
        state = discretize_state(obs)
        episode_reward, steps, done = 0.0, 0, False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = q_table[state]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action = int(random.choice(best_actions))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs)
            done = terminated or truncated

            best_next_q = np.max(q_table[next_state]) if not done else 0.0
            td_target = reward + gamma * best_next_q
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            episode_reward += reward
            steps += 1

        rewards_history.append(episode_reward)
        lengths_history.append(steps)
        epsilon_history.append(epsilon)
        best_episode_reward = max(best_episode_reward, episode_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (episode + 1) % 10 == 0:
            avg_rew = np.mean(rewards_history[-50:])
            pbar.set_postfix({
                "śr_50": f"{avg_rew:.0f}",
                "max": f"{best_episode_reward:.0f}",
                "eps": f"{epsilon:.3f}",
                "states": len(q_table),
            })

    env.close()

    # Zapis wyników
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "length", "epsilon"])
        for i, (r, l, e) in enumerate(zip(rewards_history, lengths_history, epsilon_history)):
            w.writerow([i, r, l, e])

    with open(os.path.join(save_dir, "qtable.pkl"), "wb") as f:
        pickle.dump(dict(q_table), f)

    summary = compute_summary(rewards_history, lengths_history, q_table)
    summary["algorithm"] = "Q-Learning"
    summary["gamma"] = gamma
    summary["alpha"] = alpha
    summary["num_episodes"] = num_episodes
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_learning_curves(rewards_history, lengths_history,
                         title=f"Q-Learning (Kung-Fu Master) γ={gamma}, α={alpha}",
                         out_path=os.path.join(save_dir, "curve.png"))

    print_summary(summary)
    return rewards_history, lengths_history, q_table, summary


def compute_summary(rewards, lengths, q_table=None):
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    n = len(rewards)
    last_quarter = rewards[3 * n // 4:] if n >= 4 else rewards
    first_quarter = rewards[: max(1, n // 4)]
    return {
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_first_25%": float(np.mean(first_quarter)),
        "mean_last_25%": float(np.mean(last_quarter)),
        "improvement": float(np.mean(last_quarter) - np.mean(first_quarter)),
        "mean_length": float(np.mean(lengths)),
        "explored_states": int(len(q_table)) if q_table is not None else None,
        "discounted_return_first_1000_steps": float(
            sum(r * (GAMMA ** i) for i, r in enumerate(rewards[:1000]))
        ),
    }


def print_summary(s):
    print("\n" + "=" * 60)
    print(f" PODSUMOWANIE: {s.get('algorithm', '')} (γ={s.get('gamma')})")
    print("=" * 60)
    print(f"  Najlepszy epizod:               {s['best_reward']:.0f} pkt")
    print(f"  Najgorszy epizod:               {s['worst_reward']:.0f} pkt")
    print(f"  Średnia (cały trening):         {s['mean_reward']:.1f} ± {s['std_reward']:.1f}")
    print(f"  Średnia (pierwsze 25% epiz.):   {s['mean_first_25%']:.1f}")
    print(f"  Średnia (ostatnie 25% epiz.):   {s['mean_last_25%']:.1f}")
    print(f"  Poprawa (last - first):         {s['improvement']:+.1f}")
    print(f"  Średnia długość epizodu:        {s['mean_length']:.0f} kroków")
    if s.get("explored_states"):
        print(f"  Zbadane stany w tabeli Q:       {s['explored_states']}")
    print(f"  Zdyskontowana suma (1000 ep):   {s['discounted_return_first_1000_steps']:.1f}")

    # Werdykt
    if s["improvement"] > 100:
        verdict = "Agent UCZY SIĘ — średnie nagrody rosną wyraźnie."
    elif s["improvement"] > 0:
        verdict = "Agent uczy się powoli, widać delikatny wzrost wyników."
    else:
        verdict = ("Agent NIE NAUCZYŁ SIĘ skutecznej polityki — "
                   "Q-Learning ma zbyt dużą przestrzeń stanów dla tej gry.")
    print(f"\n  Wniosek: {verdict}")
    print("=" * 60 + "\n")


def plot_learning_curves(rewards, lengths, title, out_path):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    episodes = np.arange(len(rewards))
    window = max(10, len(rewards) // 20)

    axes[0].plot(episodes, rewards, alpha=0.3, color="#1f77b4", label="surowa")
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window - 1, len(rewards)), ma,
                     color="red", linewidth=2, label=f"średnia krocząca ({window})")
    axes[0].set_xlabel("Epizod")
    axes[0].set_ylabel("Suma nagród")
    axes[0].set_title("Krzywa uczenia")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()

    axes[1].plot(episodes, lengths, alpha=0.3, color="green", label="surowa")
    if len(lengths) >= window:
        ma_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
        axes[1].plot(np.arange(window - 1, len(lengths)), ma_l,
                     color="darkgreen", linewidth=2, label=f"średnia krocząca ({window})")
    axes[1].set_xlabel("Epizod")
    axes[1].set_ylabel("Liczba kroków")
    axes[1].set_title("Długość epizodów (przeżywalność)")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    train_q_learning()
