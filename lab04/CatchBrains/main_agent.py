import gymnasium as gym

import CatchBrains  # noqa: F401
from stable_baselines3 import PPO
import numpy as np
def main():
    env = gym.make("CatchBrains/CatchBrains", render_mode=None)
    model = PPO.load("zombie_agent_v4")
    N = 100
    wins = 0
    scores = []
    lives = []
    lives_when_won = []
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1:2d}  reward={total_reward:8.1f}  score={info['score']:.0f}  lives={info['lives']}")
        if info['score'] >= 100:
            wins += 1
            lives_when_won.append(info['lives'])
        scores.append(info['score']) 
        lives.append(info['lives'])
    env.close()

    print(f"Liczba wygranych: {wins}, średni score: {np.sum(scores) / N}, średnia liczba pozostałych żyć: {np.sum(lives)/N}, średnia liczba pozostałych żyć, gdy wygrał: {np.sum(lives_when_won)/wins}")

if __name__ == "__main__":
    main()
