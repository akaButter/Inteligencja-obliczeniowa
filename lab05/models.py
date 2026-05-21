import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time

class RewardLoggerCallback(BaseCallback):
    """Zbiera nagrody epizodów w pamięci wraz z globalnym krokiem czasowym."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                self.episode_timesteps.append(self.num_timesteps)
        return True


SCRIPT_DIR = os.path.dirname(__file__) if "__file__" in locals() else "."
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "lunar")


def train_sac(total_timesteps=60_000, params=None, n_runs=10, save_dir=DEFAULT_RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    
    all_runs_timesteps = []
    all_runs_rewards = []
    
    run_times = []
    time_per_step_list = []
    time_per_episode_list = []
    
    all_episodes_rewards_flat = []
    all_episodes_lengths_flat = []

    print(f"\n[SAC] Rozpoczęcie eksperymentu: {n_runs} uruchomień po {total_timesteps} kroków.")
    print(f"Parametry: {params}")

    for run in range(n_runs):
        print(f" -> Uruchomienie {run + 1}/{n_runs} (seed={run})...")
        
        env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
        env.reset(seed=run)
        env = Monitor(env, os.path.join(save_dir, f"monitor_run_{run}.csv"))

        model = SAC(
            "MlpPolicy", env,
            device="cpu",
            seed=run,
            **params
        )

        cb = RewardLoggerCallback()
        
        start_time = time.perf_counter()
        
        try:
            model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
        except ImportError:
            model.learn(total_timesteps=total_timesteps, callback=cb)
            
        end_time = time.perf_counter()
        
        run_duration = end_time - start_time
        run_times.append(run_duration)
        
        n_steps_done = model.num_timesteps
        n_eps_done = len(cb.episode_rewards)
        
        time_per_step = run_duration / n_steps_done if n_steps_done > 0 else 0
        time_per_step_list.append(time_per_step)
        
        if n_eps_done > 0:
            time_per_episode = run_duration / n_eps_done
            time_per_episode_list.append(time_per_episode)
        
        env.close()

        if cb.episode_rewards:
            all_runs_timesteps.append(cb.episode_timesteps)
            all_runs_rewards.append(cb.episode_rewards)
            all_episodes_rewards_flat.extend(cb.episode_rewards)
            all_episodes_lengths_flat.extend(cb.episode_lengths)
            
        with open(os.path.join(save_dir, f"history_run_{run}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", "length", "global_step"])
            for idx, (r, l, t) in enumerate(zip(cb.episode_rewards, cb.episode_lengths, cb.episode_timesteps)):
                w.writerow([idx, r, l, t])

    common_timesteps = np.linspace(0, total_timesteps, 500)
    interpolated_rewards = []

    for t_seq, r_seq in zip(all_runs_timesteps, all_runs_rewards):
        full_t = [0] + list(t_seq)
        full_r = [0.0] + list(r_seq)
        interp_r = np.interp(common_timesteps, full_t, full_r)
        interpolated_rewards.append(interp_r)

    interpolated_rewards = np.array(interpolated_rewards)
    
    mean_curve = np.mean(interpolated_rewards, axis=0)
    std_curve = np.std(interpolated_rewards, axis=0)

    summary = compute_summary(all_episodes_rewards_flat, all_episodes_lengths_flat)
    summary["algorithm"] = "SAC"
    summary["total_timesteps_per_run"] = total_timesteps
    summary["n_runs"] = n_runs
    
    summary["avg_total_duration_sec"] = float(np.mean(run_times))
    summary["avg_time_per_timestep_sec"] = float(np.mean(time_per_step_list))
    summary["avg_time_per_episode_sec"] = float(np.mean(time_per_episode_list)) if time_per_episode_list else 0.0

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    
    return common_timesteps, mean_curve, std_curve, summary


def evaluate_model(model_path, n_eval_episodes=10, continuous=True):
    """Wczytuje zapisany model i odpala ewaluację."""
    env = gym.make("LunarLander-v3", continuous=continuous, render_mode="rgb_array")
    model = SAC.load(model_path)

    rewards = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += r
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"  Ep {ep+1}: reward={ep_reward:.1f}")

    env.close()
    print(f"\n[EVAL SAC] Średnia z {n_eval_episodes} ep.: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    return rewards


def compute_summary(rewards, lengths):
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    n = len(rewards)
    if n == 0:
        return {"n_episodes": 0}
    last_q = rewards[3 * n // 4:]
    first_q = rewards[: max(1, n // 4)]
    return {
        "total_tracked_episodes": int(n),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_first_25%": float(np.mean(first_q)),
        "mean_last_25%": float(np.mean(last_q)),
        "improvement": float(np.mean(last_q) - np.mean(first_q)),
        "mean_length": float(np.mean(lengths)),
        "solved_episodes": int(np.sum(rewards > 200)),
        "solved_pct": float(100 * np.sum(rewards > 200) / n),
    }


def print_summary(s):
    print("\n" + "=" * 60)
    print(f" ZAGREGOWANE PODSUMOWANIE EKSPERYMENTU ({s.get('n_runs')} Uruchomień)")
    print("=" * 60)
    print(f"  Łączna liczba epizodów (wszystkie runy): {s['total_tracked_episodes']}")
    print(f"  Najlepszy wynik (pojedynczy):           {s['best_reward']:.1f} pkt")
    print(f"  Średnia ze wszystkich uruchomień:       {s['mean_reward']:.1f} ± {s['std_reward']:.1f}")
    print(f"  Średnia długość epizodu:                {s['mean_length']:.0f} kroków")
    print(f"  Procent udanych lądowań (nagroda>200):  {s['solved_pct']:.1f}%")
    print("-" * 60)
    print(f"  [CZAS] Średni czas jednego runu:        {s['avg_total_duration_sec']:.2f} s")
    print(f"  [CZAS] Średni czas kroku (timestep):    {s['avg_time_per_timestep_sec']:.6f} s")
    print(f"  [CZAS] Średni czas jednego epizodu:     {s['avg_time_per_episode_sec']:.4f} s")
    print("=" * 60 + "\n")


def plot_aggregated_curves(results_dict, out_path):
    """Generuje jeden wspólny wykres dla wszystkich konfiguracji z cieniowaniem STD."""
    plt.figure(figsize=(10, 6))
    
    for label, data in results_dict.items():
        timesteps = data["timesteps"]
        mean = data["mean"]
        std = data["std"]
        
        line = plt.plot(timesteps, mean, label=label, linewidth=2)
        color = line[0].get_color()
        plt.fill_between(timesteps, mean - std, mean + std, color=color, alpha=0.15)
        
    plt.axhline(200, color="green", linestyle="--", alpha=0.7, label="Próg sukcesu (200)")
    plt.title("Krzywe uczenia SAC na LunarLander-v3 (Średnia ± STD z 10 uruchomień)")
    plt.xlabel("Kroki czasowe (Timesteps)")
    plt.ylabel("Suma nagród (Reward)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.show()


if __name__ == "__main__":
    params_configs = [
        {"learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005, "batch_size": 256, "buffer_size": 300_000, "target_entropy": "auto"},
        {"learning_rate": 1e-3, "gamma": 0.99, "tau": 0.01,  "batch_size": 512, "buffer_size": 300_000, "target_entropy": "auto"},
        {"learning_rate": 1e-4, "gamma": 0.995, "tau": 0.005, "batch_size": 256, "buffer_size": 300_000, "target_entropy": "auto"},
        {"learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005, "batch_size": 256, "buffer_size": 300_000, "target_entropy": -2.0}
    ]
    
    all_results = {}
    
    for i, params in enumerate(params_configs):
        cfg_label = f"Config {i}: LR={params['learning_rate']}, G={params['gamma']}"
        output_directory = os.path.join(SCRIPT_DIR, "results", f"lunar_config_{i}")
        
        timesteps, mean_curve, std_curve, _ = train_sac(
            total_timesteps=60_000, 
            params=params, 
            n_runs=10, 
            save_dir=output_directory
        )
        
        all_results[cfg_label] = {
            "timesteps": timesteps,
            "mean": mean_curve,
            "std": std_curve
        }

    global_chart_path = os.path.join(SCRIPT_DIR, "results", "comparison_curve.png")
    plot_aggregated_curves(all_results, global_chart_path)
    print(f"[SUKCES] Wykres porównawczy został zapisany w: {global_chart_path}")