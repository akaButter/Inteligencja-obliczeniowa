import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


SCRIPT_DIR = os.path.dirname(__file__) or "."
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "results", "lunar_ppo", "model")


def evaluate(model_path, n_episodes=20, render=False, continuous=True):
    render_mode = "human" if render else "rgb_array"
    env = gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode)
    model = PPO.load(model_path)

    rewards, lengths, solved = [], [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            steps += 1
            done = term or trunc
        rewards.append(ep_reward)
        lengths.append(steps)
        if ep_reward > 200:
            solved += 1
        print(f"  Ep {ep+1:>2}: reward={ep_reward:7.1f}  steps={steps:4d}  "
              f"{'OK' if ep_reward > 200 else '..' if ep_reward > 0 else 'X'}")
    env.close()

    rewards = np.array(rewards)
    print("\n" + "=" * 60)
    print(" PODSUMOWANIE EWALUACJI (deterministic policy)")
    print("=" * 60)
    print(f"  Liczba epizodów testowych:      {n_episodes}")
    print(f"  Średnia nagroda:                {rewards.mean():.1f} ± {rewards.std():.1f}")
    print(f"  Najlepszy / Najgorszy:          {rewards.max():.1f} / {rewards.min():.1f}")
    print(f"  Mediana nagrody:                {np.median(rewards):.1f}")
    print(f"  Udane lądowania (>200):         {solved}/{n_episodes} ({100*solved/n_episodes:.1f}%)")
    print(f"  Średnia długość epizodu:        {np.mean(lengths):.0f} kroków")
    if rewards.mean() > 200:
        print("\n  Wniosek: Model radzi sobie BARDZO DOBRZE - rozwiązuje zadanie.")
    elif rewards.mean() > 0:
        print("\n  Wniosek: Model jest częściowo wytrenowany - wymaga dłuższego treningu.")
    else:
        print("\n  Wniosek: Model NIE jest wytrenowany - rozbija się większość czasu.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="ścieżka do modelu PPO bez .zip "
                             "(domyślnie results/lunar_ppo/model)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true",
                        help="wyświetl okno gry (wolniejsze)")
    args = parser.parse_args()

    if not os.path.exists(args.model + ".zip"):
        raise SystemExit(f"Nie znaleziono modelu '{args.model}.zip'. "
                         f"Uruchom najpierw model_lunar.py")
    evaluate(args.model, n_episodes=args.episodes, render=args.render)
